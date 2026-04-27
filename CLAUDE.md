# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TFG (Bachelor's Thesis) project benchmarking dataset storage formats (ImageFolder, HDF5, NPZ, TFRecord) for machine learning pipelines using MNIST and Tiny ImageNet datasets. Measures load time, training efficiency, and model accuracy across formats.

## Environment Setup

```bash
source tfg_environment/bin/activate   # Must activate before running anything
```

Python 3.9.6 with TensorFlow 2.20.0, Keras 3.10.0, PyTorch 2.8.0, h5py, NumPy, Pillow, Matplotlib.

## Running Experiments

```bash
python MNIST_main.py             # MNIST pipeline: convert → validate → train → visualize
python TINYIMAGENET_main.py      # Tiny ImageNet pipeline: convert → validate → train → benchmark
```

All scripts must be run from the project root — file paths are relative.

## Architecture

Two parallel pipelines sharing the same structure:

**MNIST pipeline** (`MNIST_*.py`):
- `MNIST_main.py` — entry point, orchestrates the full pipeline
- `MNIST_dataset_converter.py` — `MNISTDataset`, `DatasetConverter` (to HDF5/NPZ/TFRecord), `DatasetValidator` (MSE comparison)
- `MNIST_neural_network_training.py` — from-scratch 2-layer NN with NumPy; loaders for each format; `NeuralNetworkMNIST` trainer
- `MNIST_visualize_results.py` — grid of predictions vs. actual labels

**Tiny ImageNet pipeline** (`TINYIMAGENET_*.py`):
- `TINYIMAGENET_main.py` — entry point; configurable via `MAX_CLASSES = 5` (up to 200)
- `TINYIMAGENET_loader.py` — `load_tinyimagenet_subset()` + format-specific loaders
- `TINYIMAGENET_dataset_module.py` — `ImageDataset`, `DatasetConverter`, `DatasetValidator`
- `TINYIMAGENET_model.py` — Keras CNN (`create_model()`): 3× Conv2D → MaxPool → Dense, input 64×64×3
- `TINYIMAGENET_visualize_results.py` — visualization helper

**Data flow:** raw image files → dataset wrapper → `DatasetConverter.to_*()` → `DatasetValidator.compare_datasets()` → model training → benchmark output.

Pre-generated binary datasets already exist under the project root (`train.h5`, `train.npz`, `train.tfrecord`, etc.) — no need to regenerate unless changing the source data or format logic.

## Key Configuration

| Script | Variable | Default | Effect |
|--------|----------|---------|--------|
| `MNIST_main.py` | `iterations` | 500 | Training iterations |
| `MNIST_main.py` | `alpha` | 0.1 | Learning rate |
| `TINYIMAGENET_main.py` | `MAX_CLASSES` | 5 | Classes used (max 200) |
| `TINYIMAGENET_main.py` | `epochs` | 10 | Training epochs |
| `TINYIMAGENET_main.py` | `batch_size` | 32 | Batch size |

## Format Implementation Notes

- **HDF5**: stores images and labels as top-level datasets via h5py
- **NPZ**: stores as NumPy compressed arrays (`images`, `labels` keys)
- **TFRecord**: serializes each sample as a `tf.train.Example`; Tiny ImageNet records also embed `height`, `width`, `channels` metadata for dynamic shape parsing
- All formats normalize pixel values to [0, 1] before training
