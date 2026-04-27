import os
import numpy as np
import time
import matplotlib.pyplot as plt
from TINYIMAGENET_dataset_module import ImageDataset, DatasetConverter, DatasetValidator
from TINYIMAGENET_loader import (
    load_tinyimagenet_subset, 
    load_from_hdf5, 
    load_from_npz, 
    load_from_tfrecord
)
from TINYIMAGENET_model import create_model
from TINYIMAGENET_visualize_results import visualize_tiny_imagenet_results

# -------------------------------
# 1. LOAD ORIGINAL DATA
# -------------------------------
root = "tiny-imagenet-200"
MAX_CLASSES = 5
print(f"📂 Loading original Tiny ImageNet subset (Classes: {MAX_CLASSES})...")

X_train_orig, Y_train_orig = load_tinyimagenet_subset(root, "train", max_classes=MAX_CLASSES)
X_val_orig, Y_val_orig = load_tinyimagenet_subset(root, "val", max_classes=MAX_CLASSES)

# Create Dataset Objects
train_dataset = ImageDataset(X_train_orig, Y_train_orig, (64, 64, 3))
val_dataset = ImageDataset(X_val_orig, Y_val_orig, (64, 64, 3))

# -------------------------------
# 2. CONVERT DATASETS
# -------------------------------
print("\n🔄 Converting datasets to binary formats...")

# Training sets
DatasetConverter.to_hdf5(train_dataset, "train_tiny.h5")
DatasetConverter.to_npz(train_dataset, "train_tiny.npz")
DatasetConverter.to_tfrecord(train_dataset, "train_tiny.tfrecord")

# Validation sets
DatasetConverter.to_hdf5(val_dataset, "val_tiny.h5")
DatasetConverter.to_npz(val_dataset, "val_tiny.npz")
DatasetConverter.to_tfrecord(val_dataset, "val_tiny.tfrecord")

# -------------------------------
# 3. VALIDATION
# -------------------------------
print("\n🔍 VALIDATING FORMAT INTEGRITY")

formats_to_validate = {
    "hdf5": (load_from_hdf5, "train_tiny.h5"),
    "npz": (load_from_npz, "train_tiny.npz"),
    "tfrecord": (load_from_tfrecord, "train_tiny.tfrecord"),
}

for name, (loader, path) in formats_to_validate.items():
    print(f"\n--- Checking {name.upper()} ---")
    X_conv, Y_conv = loader(path)
    ds_conv = ImageDataset(X_conv, Y_conv, (64, 64, 3))
    
    DatasetValidator.compare_datasets(
        train_dataset, 
        ds_conv, 
        num_samples=5, 
        visualize=False 
    )

# -------------------------------
# 4. TRAIN AND EVALUATE ON ALL FORMATS (4 Experiments)
# -------------------------------
print("\n🚀 STARTING TRAINING BENCHMARKS")

# Define paths: 'imagefolder' uses the root dir; others use file paths.
formats_training = {
    "imagefolder": (root, root),
    "hdf5": ("train_tiny.h5", "val_tiny.h5"),
    "npz": ("train_tiny.npz", "val_tiny.npz"),
    "tfrecord": ("train_tiny.tfrecord", "val_tiny.tfrecord"),
}

loaders = {
    "hdf5": load_from_hdf5,
    "npz": load_from_npz,
    "tfrecord": load_from_tfrecord
}

num_classes = len(np.unique(Y_train_orig))
results_summary = {}

for fmt, (train_path, val_path) in formats_training.items():
    print(f"\n" + "="*45)
    print(f"EXPERIMENT: {fmt.upper()} FORMAT")
    print("="*45)

    # --- Step A: Loading & Preprocessing ---
    start_load = time.time()
    
    if fmt == "imagefolder":
        # Load directly from JPG/PNG files using the loader utility
        X_train_raw, Y_train_raw = load_tinyimagenet_subset(train_path, "train", max_classes=MAX_CLASSES)
        X_val_raw, Y_val_raw = load_tinyimagenet_subset(val_path, "val", max_classes=MAX_CLASSES)
    else:
        # Load from single binary files
        X_train_raw, Y_train_raw = loaders[fmt](train_path)
        X_val_raw, Y_val_raw = loaders[fmt](val_path)
    
    # Wrap and Normalize
    ds_train = ImageDataset(X_train_raw, Y_train_raw, (64, 64, 3))
    ds_val = ImageDataset(X_val_raw, Y_val_raw, (64, 64, 3))
    ds_train.normalize()
    ds_val.normalize()
    
    load_time = time.time() - start_load
    print(f"⏱️  Loading & Normalization Time: {load_time:.4f}s")

    # --- Step B: Model Training ---
    # Create a fresh model for each format to ensure an unbiased accuracy check
    model = create_model(num_classes)

    print(f"Training on {fmt} dataset...")
    history = model.fit(
        ds_train.X, ds_train.Y,
        epochs=10, 
        batch_size=32,
        validation_data=(ds_val.X, ds_val.Y),
        verbose=1
    )

    # --- Step C: Evaluation ---
    loss, acc = model.evaluate(ds_val.X, ds_val.Y, verbose=0)
    print(f"🏆 Final Accuracy for {fmt}: {acc:.4f}")
    
    # Store results for final comparison
    results_summary[fmt] = {
        "load_time": load_time,
        "accuracy": acc,
        "file_size_mb": os.path.getsize(train_path) / (1024*1024) if fmt != "imagefolder" else 0
    }

    # --- Step D: Visualization ---
    print(f"Generating visual results for {fmt}...")
    #visualize_tiny_imagenet_results(model, ds_val, num_samples=10)

# -------------------------------
# 5. FINAL EFFICIENCY COMPARISON
# -------------------------------
print("\n" + "!"*45)
print("📊 FINAL BENCHMARK SUMMARY")
print("!"*45)
print(f"{'Format':<15} | {'Load Time (s)':<15} | {'Val Accuracy':<12}")
print("-" * 50)
for fmt, stats in results_summary.items():
    print(f"{fmt:<15} | {stats['load_time']:<15.4f} | {stats['accuracy']:<12.4f}")

print("\n✅ ALL EXPERIMENTS DONE")