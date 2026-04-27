import os
import numpy as np
from MNIST_dataset_converter import DatasetValidator, MNISTDataset, DatasetConverter
from MNIST_visualize_results import visualize_test_predictions


from MNIST_neural_network_training import (
    load_from_imagefolder,
    load_from_hdf5,
    load_from_npz,
    load_from_tfrecord,
    NeuralNetworkMNIST
)

# -------------------------------
# 1. LOAD ORIGINAL DATA
# -------------------------------
train_dir = "mnist_imagefolder/train"
test_dir = "mnist_imagefolder/test"

X_train, Y_train = load_from_imagefolder(train_dir)
X_test, Y_test = load_from_imagefolder(test_dir)

train_dataset = MNISTDataset(X_train, Y_train)
test_dataset = MNISTDataset(X_test, Y_test)

# -------------------------------
# 2. CONVERT DATASETS
# -------------------------------
print("\n🔄 Converting datasets...")

DatasetConverter.to_hdf5(train_dataset, "train.h5")
DatasetConverter.to_hdf5(test_dataset, "test.h5")

DatasetConverter.to_npz(train_dataset, "train.npz")
DatasetConverter.to_npz(test_dataset, "test.npz")

DatasetConverter.to_tfrecord(train_dataset, "train.tfrecord")
DatasetConverter.to_tfrecord(test_dataset, "test.tfrecord")

# -------------------------------
# 3. VALIDATION
# -------------------------------
print("\n🔍 VALIDATION STARTED")

formats = {
    "hdf5": (load_from_hdf5, "train.h5"),
    "npz": (load_from_npz, "train.npz"),
    "tfrecord": (load_from_tfrecord, "train.tfrecord"),
}

for name, (loader, path) in formats.items():
    print(f"\n--- Validating {name.upper()} ---")

    X_conv, Y_conv = loader(path)
    ds_conv = MNISTDataset(X_conv, Y_conv)

    DatasetValidator.compare_datasets(
        train_dataset,
        ds_conv,
        num_samples=10,
        visualize=True
    )

# -------------------------------
# 4. TRAIN ON ALL FORMATS
# -------------------------------
print("\n🚀 TRAINING ON ALL FORMATS")

formats_training = {
    "imagefolder": ("mnist_imagefolder/train", "mnist_imagefolder/test"),
    "hdf5": ("train.h5", "test.h5"),
    "npz": ("train.npz", "test.npz"),
    "tfrecord": ("train.tfrecord", "test.tfrecord"),
}

results = {}

for fmt, (train_path, test_path) in formats_training.items():
    print(f"\n==============================")
    print(f"Training with format: {fmt}")
    print(f"==============================")

    nn = NeuralNetworkMNIST(train_path, test_path, format=fmt)
    nn.load_data()
    nn.preprocess_data()
    nn.train(iterations=500, alpha=0.1)

    print("Evaluating...")
    nn.evaluate_on_test_set()
    # Optionally visualize some predictions
    visualize_test_predictions(nn, num_samples=20)

# -------------------------------
# DONE
# -------------------------------
print("\n✅ ALL DONE")