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
  python benchmark.py                        # both datasets
  python benchmark.py --dataset mnist
  python benchmark.py --dataset tinyimagenet --max-classes 5 --epochs 10
"""

import os
import time
import tracemalloc

import numpy as np

from universal_pipeline import load, save, validate, detect_format, ALL_FORMATS, Dataset

# ── Dataset-specific training imports ────────────────────────────────────────
from MNIST_neural_network_training import (
    gradient_descent, forward_prop, get_predictions, get_accuracy,
)
from TINYIMAGENET_model import create_model


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    sep = "─" * (sum(widths) + 2 * len(widths) + 2)
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))
    print(sep)


def _section(title):
    print(f"\n{'═' * 75}\n  {title}\n{'═' * 75}")


def _run_pipeline(src_train, src_val, src_fmt, out_dir, load_kwargs=None):
    """
    Shared steps 1-3 for any dataset:
      1. Load source
      2. Convert to all other formats
      3. Validate each conversion
    Returns (ds_train_src, ds_val_src, convert_times, fmt_paths).
    """
    load_kwargs = load_kwargs or {}
    os.makedirs(out_dir, exist_ok=True)
    other_fmts = [f for f in ALL_FORMATS if f != src_fmt]

    # 1. Load
    print(f"\n[1/4] Loading source ({src_fmt})...")
    t0 = time.time()
    ds_train = load(src_train, src_fmt, **load_kwargs)
    ds_val   = load(src_val,   src_fmt, **load_kwargs)
    print(f"  Train : {ds_train.num_samples}  |  Val : {ds_val.num_samples}"
          f"  |  Shape : {ds_train.img_shape}  |  Classes : {ds_train.num_classes}")

    # 2. Convert
    print(f"\n[2/4] Converting to all formats...")
    convert_times = {src_fmt: 0.0}
    for fmt in other_fmts:
        t0 = time.time()
        save(ds_train, _dst(out_dir, "train", fmt), fmt)
        save(ds_val,   _dst(out_dir, "val",   fmt), fmt)
        convert_times[fmt] = time.time() - t0
        print(f"  → {fmt:<13} {convert_times[fmt]:.3f}s")

    # 3. Validate
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
    """
    Flatten images to (784, N), run gradient descent, return test accuracy.
    Works regardless of whether X is stored as (N,28,28,1) or (N,784).
    """
    X_tr = ds_train.X.reshape(ds_train.num_samples, -1).T / 255.0
    X_va = ds_val.X.reshape(ds_val.num_samples, -1).T / 255.0

    perm = np.random.permutation(ds_train.num_samples)
    X_tr, Y_tr = X_tr[:, perm], ds_train.Y[perm]

    w1, b1, w2, b2 = gradient_descent(X_tr, Y_tr, iterations, alpha)
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, X_va)
    return get_accuracy(get_predictions(a2), ds_val.Y)


def run_mnist_benchmark(
    train_path: str = "mnist_imagefolder/train",
    val_path:   str = "mnist_imagefolder/test",
    output_dir: str = "bench_mnist",
    iterations: int = 500,
    alpha:      float = 0.1,
) -> dict:
    src_fmt = detect_format(train_path)
    _section(f"MNIST BENCHMARK  |  source: {src_fmt.upper()}")

    ds_train_src, ds_val_src, convert_times, fmt_paths = _run_pipeline(
        train_path, val_path, src_fmt, output_dir,
    )

    # 4. Train
    print(f"\n[4/4] Training on each format ({iterations} iterations)...")
    results = {}

    for fmt in ALL_FORMATS:
        t_path, v_path = fmt_paths[fmt]
        load_kw = {} if fmt != "imagefolder" else {}
        print(f"\n  [{fmt.upper()}]", end=" ", flush=True)

        t0 = time.time()
        (ds_tr, ram_tr) = _peak_ram_mb(load, t_path, fmt, **load_kw)
        (ds_va, ram_va) = _peak_ram_mb(load, v_path, fmt, **load_kw)
        load_s = time.time() - t0

        extras = [t_path + ".meta.json", v_path + ".meta.json"] if fmt == "tfrecord" else []
        file_mb = _size_mb(t_path, v_path, *extras)

        t0  = time.time()
        acc = _train_mnist(ds_tr, ds_va, iterations, alpha)
        train_s = time.time() - t0

        results[fmt] = {
            "convert_s": convert_times.get(fmt, 0.0),
            "load_s":    load_s,
            "ram_mb":    ram_tr + ram_va,
            "file_mb":   file_mb,
            "train_s":   train_s,
            "accuracy":  acc,
        }
        print(f"load={load_s:.2f}s  train={train_s:.2f}s  acc={acc:.4f}")

    _section("MNIST RESULTS")
    _print_table(
        [[fmt + (" *" if fmt == src_fmt else ""),
          f"{r['convert_s']:.3f}" if r["convert_s"] else "—",
          f"{r['load_s']:.3f}", f"{r['ram_mb']:.1f}", f"{r['file_mb']:.1f}",
          f"{r['train_s']:.3f}", f"{r['accuracy']:.4f}"]
         for fmt, r in ((f, results[f]) for f in ALL_FORMATS)],
        ["Format", "Convert(s)", "Load(s)", "RAM(MB)", "File(MB)", "Train(s)", "Accuracy"],
    )
    print("  * source format")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tiny ImageNet — Keras CNN  (TINYIMAGENET_model.create_model)
# ─────────────────────────────────────────────────────────────────────────────

def _train_tinyimagenet(ds_train: Dataset, ds_val: Dataset,
                        epochs: int = 10, batch_size: int = 32) -> float:
    """Normalise, build the existing Keras CNN, train, return val accuracy."""
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
    train_path:  str = "tiny-imagenet-200/train",
    val_path:    str = "tiny-imagenet-200/train",
    max_classes: int = 5,
    output_dir:  str = "bench_tinyimagenet",
    epochs:      int = 10,
    batch_size:  int = 32,
) -> dict:
    src_fmt = detect_format(train_path)
    _section(f"TINY IMAGENET BENCHMARK  |  {max_classes} classes  |  source: {src_fmt.upper()}")

    load_kw = {"max_classes": max_classes} if src_fmt == "imagefolder" else {}

    ds_train_src, ds_val_src, convert_times, fmt_paths = _run_pipeline(
        train_path, val_path, src_fmt, output_dir, load_kw,
    )

    # 4. Train
    print(f"\n[4/4] Training on each format ({epochs} epochs)...")
    results = {}

    for fmt in ALL_FORMATS:
        t_path, v_path = fmt_paths[fmt]
        fmt_load_kw = {"max_classes": max_classes} if fmt == "imagefolder" else {}
        print(f"\n  [{fmt.upper()}]")

        t0 = time.time()
        (ds_tr, ram_tr) = _peak_ram_mb(load, t_path, fmt, **fmt_load_kw)
        (ds_va, ram_va) = _peak_ram_mb(load, v_path, fmt, **fmt_load_kw)
        load_s = time.time() - t0

        extras = [t_path + ".meta.json", v_path + ".meta.json"] if fmt == "tfrecord" else []
        file_mb = _size_mb(t_path, v_path, *extras)

        t0  = time.time()
        acc = _train_tinyimagenet(ds_tr, ds_va, epochs, batch_size)
        train_s = time.time() - t0

        results[fmt] = {
            "convert_s": convert_times.get(fmt, 0.0),
            "load_s":    load_s,
            "ram_mb":    ram_tr + ram_va,
            "file_mb":   file_mb,
            "train_s":   train_s,
            "accuracy":  acc,
        }
        print(f"  load={load_s:.2f}s  train={train_s:.2f}s  acc={acc:.4f}")

    _section(f"TINY IMAGENET RESULTS  ({max_classes} classes)")
    _print_table(
        [[fmt + (" *" if fmt == src_fmt else ""),
          f"{r['convert_s']:.3f}" if r["convert_s"] else "—",
          f"{r['load_s']:.3f}", f"{r['ram_mb']:.1f}", f"{r['file_mb']:.1f}",
          f"{r['train_s']:.3f}", f"{r['accuracy']:.4f}"]
         for fmt, r in ((f, results[f]) for f in ALL_FORMATS)],
        ["Format", "Convert(s)", "Load(s)", "RAM(MB)", "File(MB)", "Train(s)", "Accuracy"],
    )
    print("  * source format")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format benchmark for MNIST and Tiny ImageNet")
    parser.add_argument("--dataset",     choices=["mnist", "tinyimagenet", "both"], default="both")
    parser.add_argument("--max-classes", type=int,   default=5,   help="Tiny ImageNet classes")
    parser.add_argument("--epochs",      type=int,   default=10,  help="Tiny ImageNet epochs")
    parser.add_argument("--iterations",  type=int,   default=500, help="MNIST iterations")
    parser.add_argument("--alpha",       type=float, default=0.1, help="MNIST learning rate")
    args = parser.parse_args()

    if args.dataset in ("mnist", "both"):
        run_mnist_benchmark(iterations=args.iterations, alpha=args.alpha)

    if args.dataset in ("tinyimagenet", "both"):
        run_tinyimagenet_benchmark(max_classes=args.max_classes, epochs=args.epochs)
