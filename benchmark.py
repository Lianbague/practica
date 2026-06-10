"""
benchmark.py
─────────────────────────────────────────────────────────────────────────────
Three-experiment benchmark comparing dataset storage formats.

  Experiment 1  –  Conversion cost
                   Full 4×3 conversion matrix (every format → every other).
                   Measures: conversion time, validation time, file size,
                   compression ratio.

  Experiment 2  –  Pure loading
                   No training.  Measures: load time, normalisation time,
                   total preparation time, read throughput (MB/s), peak RAM.

  Experiment 3  –  Training with a fixed CNN (fullload or streaming)
                   Measures: first-epoch time, average epoch time (epochs 2-N),
                   total training time, training throughput (imgs/s),
                   final validation accuracy and loss.
                   Streaming also records first-batch latency and peak RAM.

Dataset
  Tiny ImageNet subsets: --classes 5 | 10 | 25 | 50 | 100
  Source: tiny-imagenet-200/train/  (imagefolder, one sub-dir per class)

Usage
─────
  python benchmark.py --experiment 1 --classes 25 --runs 3 --cooldown 60
  python benchmark.py --experiment 2 --classes 25 --runs 5 --cooldown 30
  python benchmark.py --experiment 3 --mode fullload  --classes 25 --runs 3 --epochs 10 --cooldown 60
  python benchmark.py --experiment 3 --mode streaming --classes 25 --runs 3 --epochs 10 --cooldown 60

  --cooldown     N   seconds to wait between format groups (thermal recovery)
  --cooldown-run N   seconds to wait between individual runs within a format
  --format       fmt run only this format (omit to run all four)
  --runs         N   repetitions per format; report mean ± std

Per-run JSON storage
  Every completed run is immediately appended as its own record to the results
  file.  This means you can run formats or individual runs in separate
  invocations and the file accumulates correctly.  Use aggregate_results.py to
  compute mean ± std across all collected records.
"""

from __future__ import annotations

# ── macOS ARM64 / pyarrow crash fix (must be before any TF import) ────────────
import sys as _sys, types as _types, os

if "pyarrow" not in _sys.modules:
    _pa = _types.ModuleType("pyarrow")
    _pa.__version__ = "0.0.0"
    _pa.__getattr__ = lambda name: _types.ModuleType(f"pyarrow.{name}")
    _sys.modules.setdefault("pyarrow", _pa)
    for _sub in ("pyarrow.lib", "pyarrow.compute", "pyarrow.ipc",
                 "pyarrow.parquet", "pyarrow.csv", "pyarrow.dataset"):
        _sys.modules.setdefault(_sub, _types.ModuleType(_sub))
    del _pa, _sub

os.environ["KERAS_BACKEND"]             = "tensorflow"
os.environ["TF_DISABLE_PLUGGABLE_DEVICE"] = "1"

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import math
import random
import shutil
import time
import tracemalloc
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np

# ── project ───────────────────────────────────────────────────────────────────
from universal_pipeline import load, save, validate, ALL_FORMATS, Dataset
from streaming_pipeline  import stream, get_dataset_info


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch timing callback  (duck-typed; works with Keras 3 without inheriting)
# ─────────────────────────────────────────────────────────────────────────────

class EpochTimingCallback:
    """Records wall-clock time for every epoch during model.fit()."""

    def __init__(self) -> None:
        self.epoch_times: list[float] = []
        self._t0: float = 0.0

    def __getattr__(self, name: str):
        return lambda *a, **kw: None

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        self.epoch_times.append(time.perf_counter() - self._t0)

    @property
    def first_epoch_s(self) -> float:
        return self.epoch_times[0] if self.epoch_times else 0.0

    @property
    def avg_epoch_2n_s(self) -> float:
        if len(self.epoch_times) <= 1:
            return self.epoch_times[-1] if self.epoch_times else 0.0
        return float(np.mean(self.epoch_times[1:]))

    def training_throughput(self, num_samples: int, epochs: int) -> float:
        total = sum(self.epoch_times)
        return (num_samples * epochs) / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_EXT = {"imagefolder": "", "hdf5": ".h5", "npz": ".npz", "tfrecord": ".tfrecord"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def _cooldown(seconds: int, label: str = "Thermal cooldown") -> None:
    if seconds <= 0:
        return
    print()
    for remaining in range(seconds, 0, -1):
        print(f"\r  ⏸  {label} … {remaining:3d}s remaining   ",
              end="", flush=True)
        time.sleep(1)
    print(f"\r  ✓  {label} complete.                         ")


def _size_mb(*paths: str) -> float:
    total = 0
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            total += sum(f.stat().st_size for f in pp.rglob("*") if f.is_file())
        elif pp.exists():
            total += pp.stat().st_size
        meta = Path(p + ".meta.json")
        if meta.exists():
            total += meta.stat().st_size
    return total / 1024 ** 2


def _dst(out_dir: str, stem: str, fmt: str) -> str:
    if fmt == "imagefolder":
        return os.path.join(out_dir, f"{stem}_imagefolder")
    return os.path.join(out_dir, f"{stem}{_EXT[fmt]}")


def _path_exists(path: str) -> bool:
    return Path(path).exists()


def _remove(path: str) -> None:
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif p.exists():
        p.unlink(missing_ok=True)
    meta = Path(path + ".meta.json")
    if meta.exists():
        meta.unlink(missing_ok=True)


def _append_to_file(filepath: str, entry: dict) -> None:
    existing: list = []
    if os.path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)
    existing.append(entry)
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  ✔ Saved → {filepath}")


def _section(title: str) -> None:
    width = max(75, len(title) + 4)
    print(f"\n{'═' * width}\n  {title}\n{'═' * width}")


def _print_table(rows: list, headers: list) -> None:
    if not rows:
        return
    all_rows = [headers] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]
    sep  = "─" * (sum(widths) + 3 * len(widths) + 1)
    line = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(sep)
    print(line.format(*headers))
    print(sep)
    for r in rows:
        print(line.format(*r))
    print(sep)


def _stats(values: list[float]) -> tuple[float, float]:
    return round(float(np.mean(values)), 4), round(float(np.std(values)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset setup  (convert imagefolder → all binary formats once)
# ─────────────────────────────────────────────────────────────────────────────

def _setup_formats(imagefolder_path: str, classes: int, out_dir: str) -> dict:
    """
    Ensure train and val files exist for all binary formats under out_dir.
    Loads from imagefolder, splits 80/20, converts.  Skips formats already present.
    Returns {fmt: (train_path, val_path)} for all 4 formats.
    imagefolder entry points to the original source directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    train_if = os.path.join(out_dir, "train_imagefolder")
    val_if   = os.path.join(out_dir, "val_imagefolder")
    fmt_paths: dict = {"imagefolder": (train_if, val_if)}

    needs = []
    for fmt in ["imagefolder", "hdf5", "npz", "tfrecord"]:
        if fmt == "imagefolder":
            if not (_path_exists(train_if) and _path_exists(val_if)):
                needs.append("imagefolder")
        else:
            t = _dst(out_dir, "train", fmt)
            v = _dst(out_dir, "val",   fmt)
            fmt_paths[fmt] = (t, v)
            if not _path_exists(t):
                needs.append(fmt)

    if needs:
        print(f"\n  Setup: loading imagefolder ({classes} classes) …")
        ds_full = load(imagefolder_path, "imagefolder", max_classes=classes)

        n     = ds_full.num_samples
        split = int(0.8 * n)
        rng   = np.random.default_rng(seed=0)
        perm  = rng.permutation(n)
        ds_tr = Dataset(ds_full.X[perm[:split]], ds_full.Y[perm[:split]],
                        ds_full.img_shape, ds_full.class_names)
        ds_va = Dataset(ds_full.X[perm[split:]], ds_full.Y[perm[split:]],
                        ds_full.img_shape, ds_full.class_names)
        print(f"  Total {n} samples  →  train {split}  |  val {n - split}")

        for fmt in needs:
            if fmt == "imagefolder":
                print(f"  Converting → imagefolder splits …", end="", flush=True)
                t0 = time.time()
                save(ds_tr, train_if, "imagefolder")
                save(ds_va, val_if,   "imagefolder")
                print(f"  {time.time() - t0:.1f}s  ({_size_mb(train_if, val_if):.1f} MB)")
            else:
                t, v = fmt_paths[fmt]
                print(f"  Converting → {fmt} …", end="", flush=True)
                t0 = time.time()
                save(ds_tr, t, fmt)
                save(ds_va, v, fmt)
                print(f"  {time.time() - t0:.1f}s  ({_size_mb(t, v):.1f} MB)")
    else:
        print(f"\n  Setup: all formats present in {out_dir}")

    return fmt_paths


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Conversion cost
# ─────────────────────────────────────────────────────────────────────────────

def experiment_1(
    dataset:      str,
    classes:      int,
    fmt_paths:    dict,
    runs:         int,
    cooldown:     int,
    cooldown_run: int,
    fmt_filter:   str | None,
    results_file: str,
) -> None:
    """
    Full 4×3 conversion matrix (every format to every other format).
    Each run is saved immediately as its own JSON record so you can run
    formats or individual runs in separate invocations.
    """
    _section(f"EXPERIMENT 1 — CONVERSION COST  |  {dataset}  |  {classes} classes  |  {runs} runs")

    all_pairs = [(s, d) for s in ALL_FORMATS for d in ALL_FORMATS if s != d]
    pairs = [(s, d) for s, d in all_pairs
             if fmt_filter is None or s == fmt_filter or d == fmt_filter]

    tmp_dir = os.path.join(os.path.dirname(fmt_paths["hdf5"][0]), "exp1_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    src_sizes: dict[str, float] = {fmt: _size_mb(*fmt_paths[fmt]) for fmt in ALL_FORMATS}

    # Accumulate in memory for the end-of-session summary table
    session_records: dict[str, list] = {}

    for pair_idx, (src_fmt, dst_fmt) in enumerate(pairs):
        if pair_idx > 0:
            _cooldown(cooldown, "Format-group cooldown")

        key = f"{src_fmt}→{dst_fmt}"
        session_records[key] = []

        src_train, src_val = fmt_paths[src_fmt]
        load_kw = {"max_classes": classes} if src_fmt == "imagefolder" else {}

        _section(f"  {src_fmt.upper()} → {dst_fmt.upper()}")

        dst_train = _dst(tmp_dir, "train", dst_fmt)
        dst_val   = _dst(tmp_dir, "val",   dst_fmt)
        dst_mb    = 0.0

        for run_idx in range(runs):
            if run_idx > 0:
                _cooldown(cooldown_run, "Run cooldown")

            run_seed = 42 + run_idx
            _set_seed(run_seed)
            print(f"    run {run_idx + 1}/{runs}  seed={run_seed}", end="  ", flush=True)

            _remove(dst_train)
            _remove(dst_val)

            # ── conversion time ────────────────────────────────────────────
            t0 = time.perf_counter()
            ds_tr = load(src_train, src_fmt, **load_kw)
            ds_va = load(src_val,   src_fmt, **load_kw)
            save(ds_tr, dst_train, dst_fmt)
            save(ds_va, dst_val,   dst_fmt)
            t_convert = time.perf_counter() - t0

            dst_mb = _size_mb(dst_train, dst_val)
            compression_ratio = dst_mb / src_sizes[src_fmt] if src_sizes[src_fmt] > 0 else 0.0

            # ── validation time ────────────────────────────────────────────
            t0 = time.perf_counter()
            ds_conv = load(dst_train, dst_fmt)
            validate(ds_tr, ds_conv, num_samples=30)
            t_validate = time.perf_counter() - t0

            print(f"convert={t_convert:.2f}s  validate={t_validate:.2f}s  "
                  f"dst={dst_mb:.1f}MB  ratio={compression_ratio:.3f}")

            record = {
                "experiment":        1,
                "dataset":           dataset,
                "classes":           classes,
                "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
                "src_fmt":           src_fmt,
                "dst_fmt":           dst_fmt,
                "conversion":        key,
                "run":               run_idx + 1,
                "seed":              run_seed,
                "src_file_mb":       round(src_sizes[src_fmt], 3),
                "dst_file_mb":       round(dst_mb, 3),
                "compression_ratio": round(compression_ratio, 4),
                "convert_s":         round(t_convert,  4),
                "validate_s":        round(t_validate, 4),
            }
            session_records[key].append(record)
            _append_to_file(results_file, record)

    # ── summary table (this session only) ────────────────────────────────────
    _section("EXPERIMENT 1 — SESSION SUMMARY")
    rows = []
    for key, recs in session_records.items():
        cvt = [r["convert_s"]  for r in recs]
        val = [r["validate_s"] for r in recs]
        r0  = recs[0]
        cv_str = (f"{np.mean(cvt):.2f} ± {np.std(cvt):.2f}"
                  if len(cvt) > 1 else f"{cvt[0]:.2f}")
        vl_str = (f"{np.mean(val):.2f} ± {np.std(val):.2f}"
                  if len(val) > 1 else f"{val[0]:.2f}")
        rows.append([key,
                     f"{r0['src_file_mb']:.1f}",
                     f"{r0['dst_file_mb']:.1f}",
                     f"{r0['compression_ratio']:.3f}",
                     cv_str, vl_str])
    _print_table(rows, ["Conversion", "Src MB", "Dst MB", "Ratio",
                        "Convert (s)", "Validate (s)"])


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Pure loading
# ─────────────────────────────────────────────────────────────────────────────

def experiment_2(
    dataset:      str,
    classes:      int,
    fmt_paths:    dict,
    runs:         int,
    cooldown:     int,
    cooldown_run: int,
    fmt_filter:   str | None,
    results_file: str,
) -> None:
    """
    For each format, loads the full dataset from disk N times.
    Each run is saved immediately as its own JSON record.
    """
    _section(f"EXPERIMENT 2 — PURE LOADING  |  {dataset}  |  {classes} classes  |  {runs} runs")

    formats = [f for f in ALL_FORMATS if fmt_filter is None or f == fmt_filter]
    session_records: dict[str, list] = {}

    for fmt_idx, fmt in enumerate(formats):
        if fmt_idx > 0:
            _cooldown(cooldown, "Format-group cooldown")

        t_path, v_path = fmt_paths[fmt]
        load_kw = {"max_classes": classes} if fmt == "imagefolder" else {}
        file_mb = _size_mb(t_path, v_path)
        session_records[fmt] = []

        print(f"\n  [{fmt.upper()}]  on-disk size = {file_mb:.1f} MB")

        for run_idx in range(runs):
            if run_idx > 0:
                _cooldown(cooldown_run, "Run cooldown")

            run_seed = 42 + run_idx
            _set_seed(run_seed)
            print(f"    run {run_idx + 1}/{runs}  seed={run_seed}", end="  ", flush=True)

            # ── load from disk ─────────────────────────────────────────────
            tracemalloc.start()
            t0 = time.perf_counter()
            ds_tr = load(t_path, fmt, **load_kw)
            ds_va = load(v_path, fmt, **load_kw)
            t_load = time.perf_counter() - t0
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            ram_mb = peak_bytes / 1024 ** 2

            # ── normalisation ──────────────────────────────────────────────
            t0 = time.perf_counter()
            _ = ds_tr.normalized()
            _ = ds_va.normalized()
            t_normalize = time.perf_counter() - t0

            t_total        = t_load + t_normalize
            throughput_mbs = file_mb / t_load if t_load > 0 else 0.0

            print(f"load={t_load:.3f}s  norm={t_normalize:.3f}s  "
                  f"total={t_total:.3f}s  {throughput_mbs:.1f} MB/s  RAM={ram_mb:.1f} MB")

            record = {
                "experiment":     2,
                "dataset":        dataset,
                "classes":        classes,
                "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fmt":            fmt,
                "run":            run_idx + 1,
                "seed":           run_seed,
                "file_mb":        round(file_mb,       3),
                "load_s":         round(t_load,        4),
                "normalize_s":    round(t_normalize,   4),
                "total_s":        round(t_total,       4),
                "throughput_mbs": round(throughput_mbs, 3),
                "ram_mb":         round(ram_mb,         3),
            }
            session_records[fmt].append(record)
            _append_to_file(results_file, record)

    # ── summary table (this session only) ────────────────────────────────────
    _section("EXPERIMENT 2 — SESSION SUMMARY")
    rows = []
    for fmt, recs in session_records.items():
        def _sv(k):
            vals = [r[k] for r in recs]
            return (f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
                    if len(vals) > 1 else f"{vals[0]:.3f}")
        rows.append([fmt, f"{recs[0]['file_mb']:.1f}",
                     _sv("load_s"), _sv("normalize_s"), _sv("total_s"),
                     _sv("throughput_mbs"), _sv("ram_mb")])
    _print_table(rows, ["Format", "MB", "Load(s)", "Norm(s)", "Total(s)", "MB/s", "RAM(MB)"])


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3a — Training, fullload
# ─────────────────────────────────────────────────────────────────────────────

def experiment_3_fullload(
    dataset:      str,
    classes:      int,
    fmt_paths:    dict,
    epochs:       int,
    batch_size:   int,
    runs:         int,
    cooldown:     int,
    cooldown_run: int,
    fmt_filter:   str | None,
    results_file: str,
) -> None:
    """
    Loads the full dataset into RAM (once per format), then trains the CNN N times.
    Splits 80/20 train/val in memory so all formats use identical data.
    Measures per-epoch timing, total training time, and training throughput.
    """
    from TINYIMAGENET_model import create_model

    _section(f"EXPERIMENT 3 (FULLLOAD) — TRAINING  |  {dataset}  |  {classes} classes  "
             f"|  {epochs} epochs  |  {runs} runs")

    formats = [f for f in ALL_FORMATS if fmt_filter is None or f == fmt_filter]
    session_records: dict[str, list] = {}

    for fmt_idx, fmt in enumerate(formats):
        if fmt_idx > 0:
            _cooldown(cooldown, "Format-group cooldown")

        t_path, v_path = fmt_paths[fmt]
        load_kw = {"max_classes": classes} if fmt == "imagefolder" else {}

        print(f"\n  [{fmt.upper()}]")

        # ── load + preprocess train split (timed, reported as single values) ─
        tracemalloc.start()
        t0 = time.perf_counter()
        ds_full = load(t_path, fmt, **load_kw)
        ds_val_raw = load(v_path, fmt, **load_kw)
        t_load = time.perf_counter() - t0
        _, peak_load = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        t0 = time.perf_counter()
        ds_tr = ds_full.normalized()
        ds_va = ds_val_raw.normalized()
        t_preprocess = time.perf_counter() - t0

        num_samples = ds_tr.num_samples
        num_classes  = ds_tr.num_classes
        session_records[fmt] = []

        print(f"  t_load={t_load:.3f}s  t_preprocess={t_preprocess:.3f}s  "
              f"train={num_samples}  val={ds_va.num_samples}  classes={num_classes}")

        for run_idx in range(runs):
            if run_idx > 0:
                _cooldown(cooldown_run, "Run cooldown")

            run_seed = 42 + run_idx
            _set_seed(run_seed)
            print(f"\n    run {run_idx + 1}/{runs}  seed={run_seed}")

            cb    = EpochTimingCallback()
            model = create_model(num_classes)

            t0 = time.perf_counter()
            history = model.fit(
                ds_tr.X, ds_tr.Y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(ds_va.X, ds_va.Y),
                callbacks=[cb],
                verbose=1,
            )
            total_train_s = time.perf_counter() - t0

            final_accuracy = float(history.history["val_accuracy"][-1])
            final_loss     = float(history.history["val_loss"][-1])
            throughput     = cb.training_throughput(num_samples, epochs)

            print(f"    1st={cb.first_epoch_s:.2f}s  "
                  f"avg2-N={cb.avg_epoch_2n_s:.2f}s  "
                  f"total={total_train_s:.2f}s  "
                  f"{throughput:.0f} imgs/s  "
                  f"acc={final_accuracy:.4f}")

            record = {
                "experiment":        3,
                "mode":              "fullload",
                "dataset":           dataset,
                "classes":           classes,
                "epochs":            epochs,
                "batch_size":        batch_size,
                "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fmt":               fmt,
                "run":               run_idx + 1,
                "seed":              run_seed,
                "num_samples_train": num_samples,
                "t_load_s":          round(t_load,       4),
                "t_preprocess_s":    round(t_preprocess, 4),
                "t_prep_total_s":    round(t_load + t_preprocess, 4),
                "peak_load_ram_mb":  round(peak_load / 1024 ** 2, 3),
                "epoch_times":       [round(t, 4) for t in cb.epoch_times],
                "first_epoch_s":     round(cb.first_epoch_s,  4),
                "avg_epoch_2n_s":    round(cb.avg_epoch_2n_s, 4),
                "total_train_s":     round(total_train_s,     4),
                "training_throughput_imgs_s": round(throughput, 2),
                "final_accuracy":    round(final_accuracy, 4),
                "final_loss":        round(final_loss,     4),
            }
            session_records[fmt].append(record)
            _append_to_file(results_file, record)

    # ── summary table (this session only) ────────────────────────────────────
    _section("EXPERIMENT 3 (FULLLOAD) — SESSION SUMMARY")
    rows = []
    for fmt, recs in session_records.items():
        def _sv(k, fmt_str=".2f"):
            vals = [r[k] for r in recs]
            return (f"{np.mean(vals):{fmt_str}} ± {np.std(vals):{fmt_str}}"
                    if len(vals) > 1 else f"{vals[0]:{fmt_str}}")
        rows.append([fmt,
                     f"{recs[0]['t_load_s']:.3f}",
                     f"{recs[0]['t_preprocess_s']:.3f}",
                     _sv("first_epoch_s"), _sv("avg_epoch_2n_s"),
                     _sv("total_train_s"), _sv("training_throughput_imgs_s", ".0f"),
                     _sv("final_accuracy", ".4f")])
    _print_table(rows, ["Format", "Load(s)", "Prep(s)", "1stEp(s)", "Avg2-N(s)",
                        "Total(s)", "imgs/s", "Val Acc"])


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3b — Training, streaming
# ─────────────────────────────────────────────────────────────────────────────

def experiment_3_streaming(
    dataset:      str,
    classes:      int,
    fmt_paths:    dict,
    epochs:       int,
    batch_size:   int,
    runs:         int,
    cooldown:     int,
    cooldown_run: int,
    fmt_filter:   str | None,
    results_file: str,
) -> None:
    """
    Streams data batch-by-batch via tf.data (nothing loaded into RAM upfront).
    Measures: first-batch latency (cold disk read), per-epoch timing,
    total training time, training throughput, peak Python-heap RAM.
    """
    from TINYIMAGENET_model import create_model

    _section(f"EXPERIMENT 3 (STREAMING) — TRAINING  |  {dataset}  |  {classes} classes  "
             f"|  {epochs} epochs  |  {runs} runs")

    # Dataset dimensions from metadata (no full load)
    t_path_ref, v_path_ref = fmt_paths["imagefolder"]
    info_tr  = get_dataset_info(t_path_ref, "imagefolder", max_classes=classes)
    info_val = get_dataset_info(v_path_ref, "imagefolder", max_classes=classes)
    train_samples    = info_tr.num_samples
    val_samples      = info_val.num_samples
    num_classes      = info_tr.num_classes
    steps_per_epoch  = math.ceil(train_samples / batch_size)
    validation_steps = math.ceil(val_samples   / batch_size)
    shuffle_buffer   = train_samples

    print(f"\n  Dataset: {train_samples} train  |  {val_samples} val  "
          f"|  {steps_per_epoch} steps/epoch  |  {num_classes} classes")

    formats = [f for f in ALL_FORMATS if fmt_filter is None or f == fmt_filter]
    session_records: dict[str, list] = {}

    for fmt_idx, fmt in enumerate(formats):
        if fmt_idx > 0:
            _cooldown(cooldown, "Format-group cooldown")

        t_path, v_path = fmt_paths[fmt]
        stream_kw = {"max_classes": classes} if fmt == "imagefolder" else {}
        file_mb   = _size_mb(t_path, v_path)
        session_records[fmt] = []

        print(f"\n  [{fmt.upper()}]  on-disk size = {file_mb:.1f} MB")

        for run_idx in range(runs):
            if run_idx > 0:
                _cooldown(cooldown_run, "Run cooldown")

            run_seed = 42 + run_idx
            _set_seed(run_seed)
            print(f"\n    run {run_idx + 1}/{runs}  seed={run_seed}")

            train_ds = stream(t_path, fmt, batch_size=batch_size, shuffle=True,
                              shuffle_buffer=shuffle_buffer, **stream_kw)
            val_ds   = stream(v_path, fmt, batch_size=batch_size, shuffle=False,
                              **stream_kw)

            # ── first-batch latency (cold read before .repeat()) ──────────
            t0 = time.perf_counter()
            for _ in train_ds.take(1):
                pass
            first_batch_s = time.perf_counter() - t0

            train_ds = train_ds.repeat()
            val_ds   = val_ds.repeat()

            cb    = EpochTimingCallback()
            model = create_model(num_classes)

            tracemalloc.start()
            t0 = time.perf_counter()
            history = model.fit(
                train_ds,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps,
                callbacks=[cb],
                verbose=1,
            )
            total_train_s = time.perf_counter() - t0
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            ram_mb = peak_bytes / 1024 ** 2

            final_accuracy = float(history.history["val_accuracy"][-1])
            final_loss     = float(history.history["val_loss"][-1])
            throughput     = cb.training_throughput(train_samples, epochs)

            print(f"    1st_batch={first_batch_s:.3f}s  "
                  f"1st_epoch={cb.first_epoch_s:.2f}s  "
                  f"avg2-N={cb.avg_epoch_2n_s:.2f}s  "
                  f"total={total_train_s:.2f}s  "
                  f"{throughput:.0f} imgs/s  "
                  f"RAM={ram_mb:.1f} MB  "
                  f"acc={final_accuracy:.4f}")

            record = {
                "experiment":        3,
                "mode":              "streaming",
                "dataset":           dataset,
                "classes":           classes,
                "epochs":            epochs,
                "batch_size":        batch_size,
                "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fmt":               fmt,
                "run":               run_idx + 1,
                "seed":              run_seed,
                "file_mb":           round(file_mb,       3),
                "num_samples_train": train_samples,
                "epoch_times":       [round(t, 4) for t in cb.epoch_times],
                "first_batch_s":     round(first_batch_s,    4),
                "first_epoch_s":     round(cb.first_epoch_s,  4),
                "avg_epoch_2n_s":    round(cb.avg_epoch_2n_s, 4),
                "total_train_s":     round(total_train_s,     4),
                "training_throughput_imgs_s": round(throughput, 2),
                "ram_mb":            round(ram_mb,     3),
                "final_accuracy":    round(final_accuracy, 4),
                "final_loss":        round(final_loss,     4),
            }
            session_records[fmt].append(record)
            _append_to_file(results_file, record)

    # ── summary table (this session only) ────────────────────────────────────
    _section("EXPERIMENT 3 (STREAMING) — SESSION SUMMARY")
    rows = []
    for fmt, recs in session_records.items():
        def _sv(k, fmt_str=".2f"):
            vals = [r[k] for r in recs]
            return (f"{np.mean(vals):{fmt_str}} ± {np.std(vals):{fmt_str}}"
                    if len(vals) > 1 else f"{vals[0]:{fmt_str}}")
        rows.append([fmt,
                     _sv("first_batch_s", ".3f"), _sv("first_epoch_s"),
                     _sv("avg_epoch_2n_s"), _sv("total_train_s"),
                     _sv("training_throughput_imgs_s", ".0f"),
                     _sv("ram_mb", ".1f"), _sv("final_accuracy", ".4f")])
    _print_table(rows, ["Format", "1stBatch(s)", "1stEp(s)", "Avg2-N(s)",
                        "Total(s)", "imgs/s", "RAM(MB)", "Val Acc"])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dataset format benchmark — 3 experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --experiment 1 --classes 25 --runs 3 --cooldown 60
  python benchmark.py --experiment 2 --classes 25 --runs 5 --cooldown 30
  python benchmark.py --experiment 3 --mode fullload  --classes 25 --runs 3 --epochs 10 --cooldown 60
  python benchmark.py --experiment 3 --mode streaming --classes 25 --runs 3 --epochs 10 --cooldown 60
        """,
    )
    parser.add_argument("--experiment",   type=int,  required=True, choices=[1, 2, 3])
    parser.add_argument("--classes",      type=int,  default=25,
                        help="Number of Tiny ImageNet classes (5/10/25/50/100)")
    parser.add_argument("--runs",         type=int,  default=3,
                        help="Repetitions per format; results reported as mean ± std")
    parser.add_argument("--epochs",       type=int,  default=10,
                        help="Training epochs (experiment 3 only)")
    parser.add_argument("--batch-size",   type=int,  default=32,
                        help="Batch size (experiment 3 only)")
    parser.add_argument("--mode",         choices=["fullload", "streaming"], default="fullload",
                        help="Training mode (experiment 3 only)")
    parser.add_argument("--cooldown",     type=int,  default=0,
                        help="Seconds to wait between format groups (thermal recovery)")
    parser.add_argument("--cooldown-run", type=int,  default=0,
                        help="Seconds to wait between individual runs within a format")
    parser.add_argument("--format",       choices=["imagefolder", "hdf5", "npz", "tfrecord"],
                        default=None,
                        help="Run only this format (omit to run all four)")
    parser.add_argument("--src",          default="tiny-imagenet-200/train",
                        help="Path to the ImageFolder source (default: tiny-imagenet-200/train)")
    args = parser.parse_args()

    dataset  = "tinyimagenet"
    out_dir  = f"bench_exp/tinyimagenet_{args.classes}classes"

    # ── result file name ──────────────────────────────────────────────────────
    if args.experiment == 3:
        results_file = (f"results_exp3_{args.mode}_"
                        f"tinyimagenet_{args.classes}classes.json")
    else:
        results_file = (f"results_exp{args.experiment}_"
                        f"tinyimagenet_{args.classes}classes.json")

    # ── setup: ensure all formats exist ──────────────────────────────────────
    _section(f"SETUP  |  {dataset}  |  {args.classes} classes  →  {out_dir}")
    fmt_paths = _setup_formats(args.src, args.classes, out_dir)

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.experiment == 1:
        experiment_1(dataset, args.classes, fmt_paths,
                     args.runs, args.cooldown, args.cooldown_run,
                     args.format, results_file)

    elif args.experiment == 2:
        experiment_2(dataset, args.classes, fmt_paths,
                     args.runs, args.cooldown, args.cooldown_run,
                     args.format, results_file)

    elif args.experiment == 3 and args.mode == "fullload":
        experiment_3_fullload(dataset, args.classes, fmt_paths,
                              args.epochs, args.batch_size,
                              args.runs, args.cooldown, args.cooldown_run,
                              args.format, results_file)

    elif args.experiment == 3 and args.mode == "streaming":
        experiment_3_streaming(dataset, args.classes, fmt_paths,
                               args.epochs, args.batch_size,
                               args.runs, args.cooldown, args.cooldown_run,
                               args.format, results_file)
