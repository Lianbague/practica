"""
Benchmark: conversion time, loading time, and training time per format.
Covers both MNIST (custom NumPy NN) and Tiny ImageNet (Keras CNN).
"""

import os
import time
import numpy as np

# ──────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────

def hline(char="─", width=55):
    print(char * width)

def section(title):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")

def row(label, value):
    print(f"  {label:<35} {value}")


# ══════════════════════════════════════════════
# MNIST BENCHMARK
# ══════════════════════════════════════════════
section("MNIST BENCHMARK")

from MNIST_neural_network_training import (
    load_from_imagefolder, load_from_hdf5, load_from_npz, load_from_tfrecord,
    gradient_descent, forward_prop, get_predictions, get_accuracy
)
from MNIST_dataset_converter import MNISTDataset, DatasetConverter

MNIST_TRAIN_DIR  = "mnist_imagefolder/train"
MNIST_TEST_DIR   = "mnist_imagefolder/test"
MNIST_FORMATS = {
    "hdf5":     ("train.h5",       "test.h5"),
    "npz":      ("train.npz",      "test.npz"),
    "tfrecord": ("train.tfrecord", "test.tfrecord"),
}
MNIST_LOADERS = {
    "hdf5":     load_from_hdf5,
    "npz":      load_from_npz,
    "tfrecord": load_from_tfrecord,
}

# ── 1. Load ImageFolder once (needed for conversion source)
print("\nLoading MNIST from ImageFolder (source)...")
t0 = time.time()
X_train_raw, Y_train_raw = load_from_imagefolder(MNIST_TRAIN_DIR)
X_test_raw,  Y_test_raw  = load_from_imagefolder(MNIST_TEST_DIR)
mnist_imagefolder_load_time = time.time() - t0
train_ds = MNISTDataset(X_train_raw, Y_train_raw)
test_ds  = MNISTDataset(X_test_raw,  Y_test_raw)

# ── 2. Conversion times
mnist_convert_times = {}
print("\nConverting MNIST datasets...")
for fmt, (train_path, _) in MNIST_FORMATS.items():
    converter = getattr(DatasetConverter, f"to_{fmt}")
    t0 = time.time()
    converter(train_ds, train_path)
    converter(test_ds,  MNIST_FORMATS[fmt][1])
    mnist_convert_times[fmt] = time.time() - t0

# ── 3. Loading times (train + test)
mnist_load_times = {"imagefolder": mnist_imagefolder_load_time}
print("\nMeasuring MNIST load times...")
for fmt, (train_path, test_path) in MNIST_FORMATS.items():
    loader = MNIST_LOADERS[fmt]
    t0 = time.time()
    loader(train_path)
    loader(test_path)
    mnist_load_times[fmt] = time.time() - t0

# ── 4. Training times
mnist_train_times = {}
mnist_accuracies  = {}
MNIST_ITERS = 500
MNIST_ALPHA = 0.1
ALL_MNIST_LOADERS = {"imagefolder": load_from_imagefolder, **MNIST_LOADERS}
ALL_MNIST_PATHS   = {
    "imagefolder": (MNIST_TRAIN_DIR, MNIST_TEST_DIR),
    **MNIST_FORMATS,
}

print("\nTraining MNIST on all formats (500 iterations each)...")
for fmt, (train_path, test_path) in ALL_MNIST_PATHS.items():
    X_tr, Y_tr = ALL_MNIST_LOADERS[fmt](train_path)
    X_te, Y_te = ALL_MNIST_LOADERS[fmt](test_path)

    # shuffle + normalise
    perm = np.random.permutation(len(X_tr))
    X_tr, Y_tr = X_tr[perm], Y_tr[perm]
    X_tr_norm = X_tr.T / 255.0
    X_te_norm = X_te.T / 255.0

    t0 = time.time()
    w1, b1, w2, b2 = gradient_descent(X_tr_norm, Y_tr, MNIST_ITERS, MNIST_ALPHA)
    mnist_train_times[fmt] = time.time() - t0

    _, _, _, a2 = forward_prop(w1, b1, w2, b2, X_te_norm)
    mnist_accuracies[fmt] = get_accuracy(get_predictions(a2), Y_te)


# ══════════════════════════════════════════════
# TINY IMAGENET BENCHMARK
# ══════════════════════════════════════════════
section("TINY IMAGENET BENCHMARK")

from TINYIMAGENET_loader import (
    load_tinyimagenet_subset, load_from_hdf5 as ti_load_hdf5,
    load_from_npz as ti_load_npz, load_from_tfrecord as ti_load_tfrecord
)
from TINYIMAGENET_dataset_module import ImageDataset, DatasetConverter as TIConverter
from TINYIMAGENET_model import create_model

TI_ROOT      = "tiny-imagenet-200"
MAX_CLASSES  = 5
TI_FORMATS   = {
    "hdf5":     ("train_tiny.h5",       "val_tiny.h5"),
    "npz":      ("train_tiny.npz",      "val_tiny.npz"),
    "tfrecord": ("train_tiny.tfrecord", "val_tiny.tfrecord"),
}
TI_LOADERS = {
    "hdf5":     ti_load_hdf5,
    "npz":      ti_load_npz,
    "tfrecord": ti_load_tfrecord,
}

# ── 1. Load ImageFolder once
print(f"\nLoading Tiny ImageNet from ImageFolder ({MAX_CLASSES} classes)...")
t0 = time.time()
X_tr_orig, Y_tr_orig = load_tinyimagenet_subset(TI_ROOT, "train", MAX_CLASSES)
X_va_orig, Y_va_orig = load_tinyimagenet_subset(TI_ROOT, "val",   MAX_CLASSES)
ti_imagefolder_load_time = time.time() - t0
ti_train_ds = ImageDataset(X_tr_orig, Y_tr_orig, (64, 64, 3))
ti_val_ds   = ImageDataset(X_va_orig, Y_va_orig, (64, 64, 3))

# ── 2. Conversion times
ti_convert_times = {}
print("\nConverting Tiny ImageNet datasets...")
for fmt, (train_path, val_path) in TI_FORMATS.items():
    converter = getattr(TIConverter, f"to_{fmt}")
    t0 = time.time()
    converter(ti_train_ds, train_path)
    converter(ti_val_ds,   val_path)
    ti_convert_times[fmt] = time.time() - t0

# ── 3. Loading times
ti_load_times = {"imagefolder": ti_imagefolder_load_time}
print("\nMeasuring Tiny ImageNet load times...")
for fmt, (train_path, val_path) in TI_FORMATS.items():
    loader = TI_LOADERS[fmt]
    t0 = time.time()
    loader(train_path)
    loader(val_path)
    ti_load_times[fmt] = time.time() - t0

# ── 4. Training times (Keras CNN, 10 epochs)
ti_train_times = {}
ti_accuracies  = {}
num_classes    = len(np.unique(Y_tr_orig))
ALL_TI_LOADERS = {"imagefolder": None, **TI_LOADERS}
ALL_TI_PATHS   = {
    "imagefolder": (TI_ROOT, TI_ROOT),
    **TI_FORMATS,
}

print("\nTraining Tiny ImageNet CNN on all formats (10 epochs each)...")
for fmt, (train_path, val_path) in ALL_TI_PATHS.items():
    if fmt == "imagefolder":
        X_tr, Y_tr = load_tinyimagenet_subset(train_path, "train", MAX_CLASSES)
        X_va, Y_va = load_tinyimagenet_subset(val_path,   "val",   MAX_CLASSES)
    else:
        X_tr, Y_tr = TI_LOADERS[fmt](train_path)
        X_va, Y_va = TI_LOADERS[fmt](val_path)

    ds_tr = ImageDataset(X_tr, Y_tr, (64, 64, 3)); ds_tr.normalize()
    ds_va = ImageDataset(X_va, Y_va, (64, 64, 3)); ds_va.normalize()

    model = create_model(num_classes)
    t0 = time.time()
    model.fit(ds_tr.X, ds_tr.Y, epochs=10, batch_size=32,
              validation_data=(ds_va.X, ds_va.Y), verbose=0)
    ti_train_times[fmt] = time.time() - t0

    _, acc = model.evaluate(ds_va.X, ds_va.Y, verbose=0)
    ti_accuracies[fmt] = acc


# ══════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════
FORMATS_ORDER = ["imagefolder", "hdf5", "npz", "tfrecord"]

section("RESULTS — MNIST")
hline()
print(f"  {'Format':<14} {'Convert(s)':<13} {'Load(s)':<12} {'Train(s)':<12} {'Test Acc'}")
hline()
for fmt in FORMATS_ORDER:
    conv  = f"{mnist_convert_times[fmt]:.3f}" if fmt in mnist_convert_times else "—"
    load  = f"{mnist_load_times[fmt]:.3f}"
    train = f"{mnist_train_times[fmt]:.3f}"
    acc   = f"{mnist_accuracies[fmt]:.4f}"
    print(f"  {fmt:<14} {conv:<13} {load:<12} {train:<12} {acc}")
hline()

section("RESULTS — TINY IMAGENET")
hline()
print(f"  {'Format':<14} {'Convert(s)':<13} {'Load(s)':<12} {'Train(s)':<12} {'Val Acc'}")
hline()
for fmt in FORMATS_ORDER:
    conv  = f"{ti_convert_times[fmt]:.3f}" if fmt in ti_convert_times else "—"
    load  = f"{ti_load_times[fmt]:.3f}"
    train = f"{ti_train_times[fmt]:.3f}"
    acc   = f"{ti_accuracies[fmt]:.4f}"
    print(f"  {fmt:<14} {conv:<13} {load:<12} {train:<12} {acc}")
hline()

print("\n✅ Benchmark complete.\n")
