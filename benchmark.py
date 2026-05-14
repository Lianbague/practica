"""
benchmark.py
─────────────────────────────────────────────────────────────────────────────
Measures conversion time, load time, RAM, file size, training time and
accuracy across all dataset formats for both MNIST and Tiny ImageNet.

Data pipeline  : universal_pipeline  (format-agnostic)
MNIST model    : custom 2-layer NumPy NN  (MNIST_neural_network_training.py)
TinyImageNet   : Keras CNN               (TINYIMAGENET_model.py)

Usage
-----
  # Quick check — all formats, one run:
  python benchmark.py --mode fullload  --dataset mnist
  python benchmark.py --mode fullload  --dataset tinyimagenet --max-classes 10
  python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50

  # Thermally isolated, reproducible — one format per invocation:
  python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format imagefolder --runs 3
  python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format hdf5        --runs 3
  python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format npz         --runs 3
  python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format tfrecord    --runs 3

  # Aggregate saved results into a summary table:
  python aggregate_results.py results_streaming_tinyimagenet_50classes.json
"""

import json
import math
import os
import random
import time
import tracemalloc

import numpy as np

from universal_pipeline import load, save, validate, detect_format, ALL_FORMATS, Dataset
from streaming_pipeline import stream, get_dataset_info, ThroughputCallback

from MNIST_neural_network_training import (
    gradient_descent, forward_prop, get_predictions, get_accuracy,
)
from TINYIMAGENET_model import create_model


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    """Fix Python, NumPy and TensorFlow random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def _auto_results_file(mode: str, dataset: str, max_classes: int) -> str:
    if dataset == "mnist":
        return f"results_{mode}_mnist.json"
    return f"results_{mode}_tinyimagenet_{max_classes}classes.json"


def _append_results(filepath: str, mode: str, dataset: str, max_classes: int,
                    run_idx: int, seed: int, results: dict):
    entry = {
        "mode":        mode,
        "dataset":     dataset,
        "max_classes": max_classes if dataset != "mnist" else None,
        "run":         run_idx,
        "seed":        seed,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results":     results,
    }
    existing = []
    if os.path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)
    existing.append(entry)
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  ✔ Saved → {filepath}  (run {run_idx + 1})")


def _peak_ram_mb(fn, *args, **kwargs):
    tracemalloc.start()
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024 ** 2


def _size_mb(*paths: str) -> float:
    total = 0
    for p in paths:
        if os.path.isdir(p):
            for dp, _, fns in os.walk(p):
                for f in fns:
                    total += os.path.getsize(os.path.join(dp, f))
        elif os.path.exists(p):
            total += os.path.getsize(p)
    return total / 1024 ** 2


_EXT = {"imagefolder": "", "hdf5": ".h5", "npz": ".npz", "tfrecord": ".tfrecord"}


def _dst(out_dir: str, split: str, fmt: str) -> str:
    if fmt == "imagefolder":
        return os.path.join(out_dir, f"{split}_imagefolder")
    return os.path.join(out_dir, f"{split}{_EXT[fmt]}")


def _print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    sep    = "─" * (sum(widths) + 2 * len(widths) + 2)
    line   = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(sep)
    print(line.format(*headers))
    print(sep)
    for r in rows:
        print(line.format(*r))
    print(sep)


def _section(title):
    print(f"\n{'═' * 75}\n  {title}\n{'═' * 75}")


def _run_pipeline(src_train, src_val, src_fmt, out_dir, load_kwargs=None):
    """
    Steps 1-3 shared by all benchmarks:
      1. Load source dataset
      2. Convert to every other format
      3. Validate each conversion
    Returns (ds_train_src, ds_val_src, convert_times, fmt_paths).
    """
    load_kwargs = load_kwargs or {}
    os.makedirs(out_dir, exist_ok=True)
    other_fmts = [f for f in ALL_FORMATS if f != src_fmt]

    print(f"\n[1/4] Loading source ({src_fmt})...")
    ds_train = load(src_train, src_fmt, **load_kwargs)
    ds_val   = load(src_val,   src_fmt, **load_kwargs)
    print(f"  Train : {ds_train.num_samples}  |  Val : {ds_val.num_samples}"
          f"  |  Shape : {ds_train.img_shape}  |  Classes : {ds_train.num_classes}")

    print(f"\n[2/4] Converting to all formats...")
    convert_times = {src_fmt: 0.0}
    for fmt in other_fmts:
        t0 = time.time()
        save(ds_train, _dst(out_dir, "train", fmt), fmt)
        save(ds_val,   _dst(out_dir, "val",   fmt), fmt)
        convert_times[fmt] = time.time() - t0
        print(f"  → {fmt:<13} {convert_times[fmt]:.3f}s")

    print(f"\n[3/4] Validating conversions...")
    for fmt in other_fmts:
        print(f"\n  ─── {fmt.upper()} ───")
        ds_conv = load(_dst(out_dir, "train", fmt), fmt)
        validate(ds_train, ds_conv, num_samples=20).print_report()

    fmt_paths = {src_fmt: (src_train, src_val)}
    for fmt in other_fmts:
        fmt_paths[fmt] = (_dst(out_dir, "train", fmt), _dst(out_dir, "val", fmt))

    return ds_train, ds_val, convert_times, fmt_paths


# ─────────────────────────────────────────────────────────────────────────────
# MNIST — custom 2-layer NumPy neural network
# ─────────────────────────────────────────────────────────────────────────────

def _train_mnist(ds_train: Dataset, ds_val: Dataset,
                 iterations: int = 500, alpha: float = 0.1) -> float:
    X_tr = ds_train.X.reshape(ds_train.num_samples, -1).T / 255.0
    X_va = ds_val.X.reshape(ds_val.num_samples, -1).T / 255.0
    perm = np.random.permutation(ds_train.num_samples)
    X_tr, Y_tr = X_tr[:, perm], ds_train.Y[perm]
    w1, b1, w2, b2 = gradient_descent(X_tr, Y_tr, iterations, alpha)
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, X_va)
    return get_accuracy(get_predictions(a2), ds_val.Y)


def run_mnist_benchmark(
    train_path:   str   = "mnist_imagefolder/train",
    val_path:     str   = "mnist_imagefolder/test",
    output_dir:   str   = "bench_mnist",
    iterations:   int   = 500,
    alpha:        float = 0.1,
    fmt_filter:   str   = None,
    runs:         int   = 1,
    seed:         int   = 42,
    results_file: str   = None,
) -> dict:
    src_fmt = detect_format(train_path)
    label   = (f"  |  format: {fmt_filter.upper()}" if fmt_filter else "") + \
              (f"  |  {runs} run(s)" if runs > 1 else "")
    _section(f"MNIST BENCHMARK  |  source: {src_fmt.upper()}{label}")

    ds_train_src, ds_val_src, convert_times, fmt_paths = _run_pipeline(
        train_path, val_path, src_fmt, output_dir,
    )

    print(f"\n[4/4] Training on each format ({iterations} iterations)...")
    formats_to_train = [fmt_filter] if fmt_filter else ALL_FORMATS
    last_results = {}

    for fmt in formats_to_train:
        t_path, v_path = fmt_paths[fmt]

        t0 = time.time()
        (ds_tr, ram_tr) = _peak_ram_mb(load, t_path, fmt)
        (ds_va, ram_va) = _peak_ram_mb(load, v_path, fmt)
        load_s = time.time() - t0

        extras  = [t_path + ".meta.json", v_path + ".meta.json"] if fmt == "tfrecord" else []
        file_mb = _size_mb(t_path, v_path, *extras)

        for run_idx in range(runs):
            run_seed = seed + run_idx
            _set_seed(run_seed)
            print(f"\n  [{fmt.upper()}]  run {run_idx + 1}/{runs}  seed={run_seed}", end="  ", flush=True)

            t0      = time.time()
            acc     = _train_mnist(ds_tr, ds_va, iterations, alpha)
            train_s = time.time() - t0

            row = {
                "convert_s": convert_times.get(fmt, 0.0),
                "load_s":    load_s,
                "ram_mb":    ram_tr + ram_va,
                "file_mb":   file_mb,
                "train_s":   train_s,
                "accuracy":  acc,
            }
            print(f"load={load_s:.2f}s  train={train_s:.2f}s  acc={acc:.4f}")
            if results_file:
                _append_results(results_file, "fullload", "mnist", 0,
                                run_idx, run_seed, {fmt: row})
            last_results[fmt] = row

    if not fmt_filter:
        _section("MNIST RESULTS")
        _print_table(
            [[fmt + (" *" if fmt == src_fmt else ""),
              f"{r['convert_s']:.3f}" if r["convert_s"] else "—",
              f"{r['load_s']:.3f}", f"{r['ram_mb']:.1f}", f"{r['file_mb']:.1f}",
              f"{r['train_s']:.3f}", f"{r['accuracy']:.4f}"]
             for fmt, r in ((f, last_results[f]) for f in ALL_FORMATS if f in last_results)],
            ["Format", "Convert(s)", "Load(s)", "RAM(MB)", "File(MB)", "Train(s)", "Accuracy"],
        )
        print("  * source format")

    return last_results


# ─────────────────────────────────────────────────────────────────────────────
# Tiny ImageNet — Keras CNN
# ─────────────────────────────────────────────────────────────────────────────

def _train_tinyimagenet(ds_train: Dataset, ds_val: Dataset,
                        epochs: int = 10, batch_size: int = 32) -> float:
    ds_tr = ds_train.normalized()
    ds_va = ds_val.normalized()
    model = create_model(ds_train.num_classes)
    model.fit(
        ds_tr.X, ds_tr.Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(ds_va.X, ds_va.Y),
        verbose=1,
    )
    _, acc = model.evaluate(ds_va.X, ds_va.Y, verbose=0)
    return float(acc)


def run_tinyimagenet_benchmark(
    train_path:   str = "tiny-imagenet-200/train",
    val_path:     str = "tiny-imagenet-200/train",
    max_classes:  int = 5,
    output_dir:   str = "bench_tinyimagenet",
    epochs:       int = 10,
    batch_size:   int = 32,
    fmt_filter:   str = None,
    runs:         int = 1,
    seed:         int = 42,
    results_file: str = None,
) -> dict:
    src_fmt = detect_format(train_path)
    label   = (f"  |  format: {fmt_filter.upper()}" if fmt_filter else "") + \
              (f"  |  {runs} run(s)" if runs > 1 else "")
    _section(f"TINY IMAGENET BENCHMARK  |  {max_classes} classes  |  source: {src_fmt.upper()}{label}")

    load_kw = {"max_classes": max_classes} if src_fmt == "imagefolder" else {}
    ds_train_src, ds_val_src, convert_times, fmt_paths = _run_pipeline(
        train_path, val_path, src_fmt, output_dir, load_kw,
    )

    print(f"\n[4/4] Training on each format ({epochs} epochs)...")
    formats_to_train = [fmt_filter] if fmt_filter else ALL_FORMATS
    last_results = {}

    for fmt in formats_to_train:
        t_path, v_path = fmt_paths[fmt]
        fmt_load_kw = {"max_classes": max_classes} if fmt == "imagefolder" else {}

        t0 = time.time()
        (ds_tr, ram_tr) = _peak_ram_mb(load, t_path, fmt, **fmt_load_kw)
        (ds_va, ram_va) = _peak_ram_mb(load, v_path, fmt, **fmt_load_kw)
        load_s = time.time() - t0

        extras  = [t_path + ".meta.json", v_path + ".meta.json"] if fmt == "tfrecord" else []
        file_mb = _size_mb(t_path, v_path, *extras)

        for run_idx in range(runs):
            run_seed = seed + run_idx
            _set_seed(run_seed)
            print(f"\n  [{fmt.upper()}]  run {run_idx + 1}/{runs}  seed={run_seed}")

            t0      = time.time()
            acc     = _train_tinyimagenet(ds_tr, ds_va, epochs, batch_size)
            train_s = time.time() - t0

            row = {
                "convert_s": convert_times.get(fmt, 0.0),
                "load_s":    load_s,
                "ram_mb":    ram_tr + ram_va,
                "file_mb":   file_mb,
                "train_s":   train_s,
                "accuracy":  acc,
            }
            print(f"  load={load_s:.2f}s  train={train_s:.2f}s  acc={acc:.4f}")
            if results_file:
                _append_results(results_file, "fullload", "tinyimagenet", max_classes,
                                run_idx, run_seed, {fmt: row})
            last_results[fmt] = row

    if not fmt_filter:
        _section(f"TINY IMAGENET RESULTS  ({max_classes} classes)")
        _print_table(
            [[fmt + (" *" if fmt == src_fmt else ""),
              f"{r['convert_s']:.3f}" if r["convert_s"] else "—",
              f"{r['load_s']:.3f}", f"{r['ram_mb']:.1f}", f"{r['file_mb']:.1f}",
              f"{r['train_s']:.3f}", f"{r['accuracy']:.4f}"]
             for fmt, r in ((f, last_results[f]) for f in ALL_FORMATS if f in last_results)],
            ["Format", "Convert(s)", "Load(s)", "RAM(MB)", "File(MB)", "Train(s)", "Accuracy"],
        )
        print("  * source format")

    return last_results


# ─────────────────────────────────────────────────────────────────────────────
# Tiny ImageNet — streaming benchmark  (batch-by-batch, no full RAM load)
# ─────────────────────────────────────────────────────────────────────────────

def run_tinyimagenet_streaming_benchmark(
    train_path:   str = "tiny-imagenet-200/train",
    val_path:     str = "tiny-imagenet-200/train",
    max_classes:  int = 5,
    output_dir:   str = "bench_tinyimagenet",
    epochs:       int = 10,
    batch_size:   int = 32,
    fmt_filter:   str = None,
    runs:         int = 1,
    seed:         int = 42,
    results_file: str = None,
) -> dict:
    src_fmt = detect_format(train_path)
    label   = (f"  |  format: {fmt_filter.upper()}" if fmt_filter else "") + \
              (f"  |  {runs} run(s)" if runs > 1 else "")
    _section(
        f"TINY IMAGENET STREAMING BENCHMARK  |  {max_classes} classes  "
        f"|  source: {src_fmt.upper()}{label}"
    )

    load_kw = {"max_classes": max_classes} if src_fmt == "imagefolder" else {}
    ds_train_src, ds_val_src, convert_times, fmt_paths = _run_pipeline(
        train_path, val_path, src_fmt, output_dir, load_kw,
    )

    print(f"\n[4/4] Streaming training on each format ({epochs} epochs)...")
    formats_to_train = [fmt_filter] if fmt_filter else ALL_FORMATS
    last_results     = {}

    steps_per_epoch  = math.ceil(ds_train_src.num_samples / batch_size)
    validation_steps = math.ceil(ds_val_src.num_samples   / batch_size)
    shuffle_buffer   = ds_train_src.num_samples

    for fmt in formats_to_train:
        t_path, v_path = fmt_paths[fmt]
        stream_kw      = {"max_classes": max_classes} if fmt == "imagefolder" else {}
        extras         = [t_path + ".meta.json", v_path + ".meta.json"] if fmt == "tfrecord" else []
        file_mb        = _size_mb(t_path, v_path, *extras)

        for run_idx in range(runs):
            run_seed = seed + run_idx
            _set_seed(run_seed)
            print(f"\n  [{fmt.upper()}]  run {run_idx + 1}/{runs}  seed={run_seed}")

            train_ds = stream(t_path, fmt, batch_size=batch_size, shuffle=True,
                              shuffle_buffer=shuffle_buffer, **stream_kw)
            val_ds   = stream(v_path, fmt, batch_size=batch_size, shuffle=False, **stream_kw)

            # First-batch latency: cold read from disk before .repeat()
            t0 = time.time()
            for _ in train_ds.take(1):
                pass
            first_batch_s = time.time() - t0

            train_ds = train_ds.repeat()
            val_ds   = val_ds.repeat()

            model         = create_model(ds_train_src.num_classes)
            throughput_cb = ThroughputCallback()

            tracemalloc.start()
            t0 = time.time()
            history = model.fit(
                train_ds,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps,
                callbacks=[throughput_cb],
                verbose=1,
            )
            train_s = time.time() - t0
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            ram_mb = peak_bytes / 1024 ** 2

            acc       = float(history.history["val_accuracy"][-1])
            samples_s = throughput_cb.throughput(ds_train_src.num_samples)

            row = {
                "convert_s":     convert_times.get(fmt, 0.0),
                "first_batch_s": first_batch_s,
                "ram_mb":        ram_mb,
                "file_mb":       file_mb,
                "train_s":       train_s,
                "samples_s":     samples_s,
                "accuracy":      acc,
            }
            print(f"  1st_batch={first_batch_s:.3f}s  train={train_s:.2f}s  "
                  f"throughput={samples_s:.0f} samples/s  acc={acc:.4f}")
            if results_file:
                _append_results(results_file, "streaming", "tinyimagenet", max_classes,
                                run_idx, run_seed, {fmt: row})
            last_results[fmt] = row

    if not fmt_filter:
        _section(f"TINY IMAGENET STREAMING RESULTS  ({max_classes} classes)")
        _print_table(
            [[fmt + (" *" if fmt == src_fmt else ""),
              f"{r['convert_s']:.3f}" if r["convert_s"] else "—",
              f"{r['first_batch_s']:.3f}",
              f"{r['ram_mb']:.1f}",
              f"{r['file_mb']:.1f}",
              f"{r['train_s']:.3f}",
              f"{r['samples_s']:.0f}",
              f"{r['accuracy']:.4f}"]
             for fmt, r in ((f, last_results[f]) for f in ALL_FORMATS if f in last_results)],
            ["Format", "Convert(s)", "1stBatch(s)", "RAM(MB)", "File(MB)",
             "Train(s)", "Samples/s", "Accuracy"],
        )
        print("  * source format")
        print("  RAM(MB): Python heap peak (tracemalloc) — excludes TensorFlow allocations")

    return last_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Format benchmark for MNIST and Tiny ImageNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick check — all formats, one run:
  python benchmark.py --mode fullload  --dataset mnist
  python benchmark.py --mode fullload  --dataset tinyimagenet --max-classes 10
  python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50

  # Thermally isolated, reproducible — one format per invocation:
  python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format imagefolder --runs 3
  python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format hdf5        --runs 3
  python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format npz         --runs 3
  python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format tfrecord    --runs 3

  # Aggregate saved results:
  python aggregate_results.py results_streaming_tinyimagenet_50classes.json
        """,
    )
    parser.add_argument(
        "--mode", required=True, choices=["fullload", "streaming"],
        help="fullload: load entire dataset into RAM first; streaming: batch-by-batch via tf.data",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["mnist", "tinyimagenet"],
        help="Which dataset to benchmark",
    )
    parser.add_argument(
        "--format", choices=["imagefolder", "hdf5", "npz", "tfrecord"], default=None,
        help="Run only this format (omit to run all formats in one go)",
    )
    parser.add_argument("--runs",        type=int,   default=1,   help="Repetitions per format (default 1)")
    parser.add_argument("--seed",        type=int,   default=42,  help="Base random seed; each run uses seed+run_idx")
    parser.add_argument("--max-classes", type=int,   default=5,   help="Tiny ImageNet: number of classes (default 5)")
    parser.add_argument("--epochs",      type=int,   default=10,  help="Tiny ImageNet: training epochs")
    parser.add_argument("--iterations",  type=int,   default=500, help="MNIST: training iterations")
    parser.add_argument("--alpha",       type=float, default=0.1, help="MNIST: learning rate")
    args = parser.parse_args()

    if args.mode == "streaming" and args.dataset == "mnist":
        parser.error("--mode streaming --dataset mnist is not yet implemented")

    results_file = _auto_results_file(args.mode, args.dataset, args.max_classes)

    if args.mode == "fullload" and args.dataset == "mnist":
        run_mnist_benchmark(
            iterations=args.iterations, alpha=args.alpha,
            fmt_filter=args.format, runs=args.runs, seed=args.seed,
            results_file=results_file,
        )

    elif args.mode == "fullload" and args.dataset == "tinyimagenet":
        run_tinyimagenet_benchmark(
            max_classes=args.max_classes, epochs=args.epochs,
            fmt_filter=args.format, runs=args.runs, seed=args.seed,
            results_file=results_file,
        )

    elif args.mode == "streaming" and args.dataset == "tinyimagenet":
        run_tinyimagenet_streaming_benchmark(
            max_classes=args.max_classes, epochs=args.epochs,
            fmt_filter=args.format, runs=args.runs, seed=args.seed,
            results_file=results_file,
        )
