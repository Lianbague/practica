# Project Documentation
## Benchmarking Dataset Storage Formats for Machine Learning Pipelines

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background Concepts](#2-background-concepts)
3. [Datasets Used](#3-datasets-used)
4. [Storage Formats Studied](#4-storage-formats-studied)
5. [System Architecture](#5-system-architecture)
6. [File-by-File Reference](#6-file-by-file-reference)
   - 6.1 [universal_pipeline.py](#61-universal_pipelinepy)
   - 6.2 [streaming_pipeline.py](#62-streaming_pipelinepy)
   - 6.3 [benchmark.py](#63-benchmarkpy)
   - 6.4 [aggregate_results.py](#64-aggregate_resultspy)
   - 6.5 [MNIST_neural_network_training.py](#65-mnist_neural_network_trainingpy)
   - 6.6 [MNIST_dataset_converter.py](#66-mnist_dataset_converterpy)
   - 6.7 [MNIST_main.py](#67-mnist_mainpy)
   - 6.8 [TINYIMAGENET_loader.py](#68-tinyimagenet_loaderpy)
   - 6.9 [TINYIMAGENET_dataset_module.py](#69-tinyimagenet_dataset_modulepy)
   - 6.10 [TINYIMAGENET_model.py](#610-tinyimagenet_modelpy)
   - 6.11 [TINYIMAGENET_main.py](#611-tinyimagenet_mainpy)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [How to Run the Experiments](#8-how-to-run-the-experiments)
9. [Understanding the Benchmark Results](#9-understanding-the-benchmark-results)
10. [Limitations and Methodological Notes](#10-limitations-and-methodological-notes)

---

## 1. Project Overview

This project is a Bachelor's Thesis (TFG) that studies how the choice of **dataset storage format** affects the efficiency of machine learning pipelines. More specifically, it answers the question:

> *Does it matter how we store our training images on disk? Is one file format faster, smaller, or more memory-efficient than another?*

To answer this question, the project:
1. Takes two well-known image datasets (MNIST and Tiny ImageNet).
2. Converts each dataset into four different file formats.
3. Measures — for each format — how long it takes to load, how much memory it uses, how large the files are on disk, and how well a model trained on that data performs.
4. Does this in two modes: **full-load** (all data into RAM first, then train) and **streaming** (data read batch by batch during training, never fully in RAM).

The experiments are fully automated: a single command runs the entire pipeline from raw images to final benchmark table.

---

## 2. Background Concepts

Before reading the code, the following concepts are necessary to understand what the project does and why.

### 2.1 What is a Dataset in Machine Learning?

A dataset for image classification consists of two things:
- **Images** (X): a collection of pictures, each represented as a grid of numbers (pixels).
- **Labels** (Y): for each image, a number indicating which category it belongs to (e.g. 0 = cat, 1 = dog).

The model learns to look at X and predict Y.

### 2.2 What is a Pixel?

A pixel is the smallest unit of an image. Each pixel has a colour value between 0 (black) and 255 (white) for grayscale images. For colour images, each pixel has three values — Red, Green, Blue — each between 0 and 255.

A 28×28 grayscale image like in MNIST is therefore a grid of 28 rows × 28 columns = 784 numbers.
A 64×64 colour image like in Tiny ImageNet is a grid of 64×64×3 = 12,288 numbers.

### 2.3 What is a NumPy Array?

NumPy is a Python library for working with large arrays of numbers efficiently. When images are loaded into memory, they are stored as NumPy arrays. For example, 60,000 MNIST images stored together form an array of shape (60000, 28, 28, 1) — meaning 60,000 images, each 28 pixels tall, 28 pixels wide, 1 colour channel.

### 2.4 What is a Neural Network?

A neural network is a mathematical model that learns to map inputs (images) to outputs (class labels) by adjusting internal parameters (called weights) through a process called training. Training means showing the model many examples repeatedly and nudging the weights in the direction that reduces prediction errors.

### 2.5 What is a Format?

In computing, a format is the structure in which data is stored in a file. The same 60,000 MNIST images can be stored as:
- Individual PNG files in folders (ImageFolder)
- A single compressed binary file (HDF5 or NPZ)
- A sequential record file (TFRecord)

The data is identical in all cases, but the files differ in size, read speed, and memory consumption.

### 2.6 What is RAM?

RAM (Random Access Memory) is the computer's working memory — the space where data lives while a program is running. When we "load a dataset", we are copying it from disk into RAM. The more data we load, the more RAM we consume.

### 2.7 What is Streaming?

Streaming means reading data piece by piece as it is needed, rather than loading everything at once. A real-world analogy: watching a film on Netflix (streaming) versus downloading the whole film first (full load). In machine learning, streaming means the model trains on one small batch of images at a time, reading each batch from disk on demand.

---

## 3. Datasets Used

### 3.1 MNIST

MNIST is one of the most famous benchmark datasets in machine learning. It contains 70,000 handwritten digit images:
- 60,000 for training, 10,000 for testing.
- Each image is 28×28 pixels in **grayscale** (one colour channel).
- There are 10 classes: digits 0 through 9.

The images are stored originally as PNG files, one per image, organised in folders named `0/`, `1/`, ... `9/`.

**Why MNIST?** It is small, fast to work with, and well-understood. Results are easy to verify and the training model is simple.

### 3.2 Tiny ImageNet

Tiny ImageNet is a reduced version of the famous ImageNet dataset. It contains:
- Up to 200 classes of everyday objects (cars, animals, food, etc.).
- 500 training images per class → up to 100,000 training images total.
- Each image is 64×64 pixels in **colour** (three channels: RGB).

In this project, a subset of classes is used (configurable via `--max-classes`, default 5) to keep experiment times manageable.

**Why Tiny ImageNet?** It is much larger and more complex than MNIST, making format differences more visible.

---

## 4. Storage Formats Studied

The project compares four formats:

### 4.1 ImageFolder

The simplest format. Images are stored as individual files (PNG or JPEG) organised in a directory tree:
```
dataset/
  class_0/
    image_001.png
    image_002.png
  class_1/
    image_001.png
    ...
```
- **Advantage**: human-readable, no conversion needed, easy to inspect visually.
- **Disadvantage**: loading requires opening thousands of individual files — very slow for large datasets. Also uses less disk space than binary formats when images are stored as compressed PNG/JPEG.

### 4.2 HDF5 (.h5)

HDF5 (Hierarchical Data Format version 5) is a binary file format designed for storing large scientific datasets. The entire dataset — all images and all labels — is stored in a single `.h5` file.
- **Advantage**: very fast to load (single binary read), supports partial/random access to the data.
- **Disadvantage**: not human-readable; requires the `h5py` library.

### 4.3 NPZ (.npz)

NPZ is NumPy's native compressed archive format. It stores arrays directly as compressed binary data in a single `.npz` file.
- **Advantage**: extremely simple, native to NumPy (no extra library needed), supports memory-mapped access (only loading pages that are actually read).
- **Disadvantage**: not designed for streaming; the entire array structure must be opened to access any part.

### 4.4 TFRecord (.tfrecord)

TFRecord is TensorFlow's native sequential record format. Each image is serialised (converted to a byte sequence) as a separate record and written one after another into a single file.
- **Advantage**: designed specifically for streaming — TensorFlow's `tf.data` pipeline reads it with parallel I/O, prefetching, and hardware acceleration.
- **Disadvantage**: requires TensorFlow to read; more complex to write; a sidecar metadata file (`.meta.json`) is needed to store class names and image shape since TFRecord has no header.

---

## 5. System Architecture

The project has two generations of code:

### 5.1 Original Pipeline (per-dataset, legacy)

The first version of the project had two completely separate and independent codebases — one for MNIST, one for Tiny ImageNet — each with its own dataset class, converter, validator, and loaders.

```
MNIST pipeline:
  MNIST_dataset_converter.py  → MNISTDataset, DatasetConverter, DatasetValidator
  MNIST_neural_network_training.py  → loaders + neural network
  MNIST_main.py  → entry point

Tiny ImageNet pipeline:
  TINYIMAGENET_loader.py  → loaders
  TINYIMAGENET_dataset_module.py  → ImageDataset, DatasetConverter, DatasetValidator
  TINYIMAGENET_model.py  → Keras CNN
  TINYIMAGENET_main.py  → entry point
```

The problem with this design is duplication: adding a new dataset or format required writing the same converter and validator logic twice (or more). The two pipelines were not interchangeable.

### 5.2 Universal Pipeline (new system)

The second version introduces a **common intermediate representation** — the `Dataset` dataclass — and a unified set of loaders and savers that work with any dataset and any format.

```
Any dataset, any format
        │
        ▼
   universal_pipeline.py
        │
   ┌────┴────┐
   │ Dataset │  ← universal in-memory container
   └────┬────┘
        │
   ┌────┴──────────────────────────────┐
   │  Four loaders  │  Four savers     │
   │  imagefolder   │  imagefolder     │
   │  hdf5          │  hdf5            │
   │  npz           │  npz             │
   │  tfrecord      │  tfrecord        │
   └───────────────────────────────────┘
        │
        ▼
   benchmark.py  ← measures all formats
        │
   ┌────┴─────────────────────────────────────┐
   │  MNIST model  │  Tiny ImageNet model      │
   │  (NumPy NN)   │  (Keras CNN)              │
   └──────────────────────────────────────────┘
```

The key insight: **every loader returns the same `Dataset` object, and every saver reads from that same object**. This means any format can be converted to any other format in two steps: load → save. The training models never interact with files — they only receive NumPy arrays.

### 5.3 Streaming Pipeline

The third component, `streaming_pipeline.py`, adds a more realistic benchmark mode where data is never fully loaded into RAM. Instead, `tf.data.Dataset` objects are returned that feed batches directly into Keras during training. This isolates the I/O cost per format.

```
streaming_pipeline.py
        │
        ▼
  tf.data.Dataset  ← yields (batch of images, batch of labels)
        │
        ▼
  model.fit(...)  ← Keras pulls batches on demand
```

---

## 6. File-by-File Reference

---

### 6.1 `universal_pipeline.py`

**Purpose**: The core of the new system. Provides format-agnostic loading, saving, conversion, and validation for any image classification dataset.

**Dependencies**: `numpy`, `Pillow` (PIL), `h5py`, `tensorflow` (lazy), `json`, `pathlib`

---

#### Class: `Dataset`

```python
@dataclass
class Dataset:
    X: np.ndarray    # shape: (N, H, W, C), dtype: uint8
    Y: np.ndarray    # shape: (N,),          dtype: int32
    img_shape: tuple # (H, W, C)
    class_names: list
```

The universal in-memory container that represents any image classification dataset. All loaders return this type; all savers accept this type.

- `X` holds all images as a single NumPy array. The shape is (N, H, W, C): N images, H pixels tall, W pixels wide, C colour channels (1 for grayscale, 3 for colour). Pixel values are integers from 0 to 255.
- `Y` holds all labels as integer class indices. Label 0 means the first class, label 1 the second, and so on.
- `img_shape` is a tuple (H, W, C) describing the dimensions of a single image.
- `class_names` is a list of human-readable class names in the same order as the label indices (e.g. `["cat", "dog"]` means label 0 = cat, label 1 = dog).

**Properties:**
- `num_samples` → returns the number of images (len of X).
- `num_classes` → returns the number of distinct classes (len of class_names).

**Method: `normalized()`**
Returns a new `Dataset` object where `X` has been divided by 255.0 and cast to float32. The original dataset is not modified. This is needed before training because neural networks expect inputs in the range [0, 1] rather than [0, 255].

---

#### Function: `detect_format(path)`

```python
def detect_format(path: str) -> str
```

Automatically determines the format of a dataset from its file path, so the user does not need to specify it explicitly.

- If the path is an existing directory → `"imagefolder"`
- If the path ends in `.h5` or `.hdf5` → `"hdf5"`
- If the path ends in `.npz` → `"npz"`
- If the path ends in `.tfrecord` → `"tfrecord"`
- If the path has no extension → `"imagefolder"` (treats it as a directory that doesn't exist yet)

---

#### Function: `_open_image(path, target_size, target_channels)`

```python
def _open_image(path, target_size=None, target_channels=None) -> np.ndarray
```

Internal helper (not part of the public API) that opens a single image file and returns it as a NumPy array of shape (H, W, C).

Key behaviours:
- RGBA images (PNG with transparency) are converted to RGB by dropping the alpha channel.
- Unusual modes (palette-based, etc.) are converted to RGB.
- If `target_channels=3` and the image is grayscale (L mode), it is converted to RGB — this handles datasets where most images are colour but a few are grayscale.
- If `target_channels=1` and the image is RGB, it is converted to grayscale.
- If `target_size` is given, the image is resized to match.
- A 2D grayscale array (H, W) is reshaped to (H, W, 1) so that all images consistently have three dimensions.

---

#### Function: `load_imagefolder(root, max_classes, target_size)`

```python
def load_imagefolder(root, max_classes=None, target_size=None) -> Dataset
```

Loads a dataset stored in the standard ImageFolder layout: one subdirectory per class, containing image files of any supported format (PNG, JPEG, BMP, TIFF, WebP).

**Process:**
1. Lists all subdirectories in `root`, sorted alphabetically. Each subdirectory name becomes a class name.
2. If `max_classes` is given, only the first N classes are loaded.
3. Iterates through every image file in every class directory (including nested subdirectories via `rglob`).
4. Opens the first successfully loaded image to lock in the spatial size (`_size`) and channel count (`_channels`). All subsequent images are normalised to match these values. This handles Tiny ImageNet, where a small number of images are grayscale despite the rest being colour.
5. Stacks all images into a single NumPy array X and all labels into Y.
6. Returns a `Dataset`.

**Important note on channel locking**: the `_channels` variable is set from the very first image. If the first image is RGB, all subsequent grayscale images will be converted to RGB. This ensures the final array X has a consistent shape — a requirement for NumPy to stack them together.

---

#### Function: `load_hdf5(path)`

```python
def load_hdf5(path: str) -> Dataset
```

Loads a dataset from a `.h5` file. The file must have been saved by `save_hdf5` (or be compatible with its structure).

**Process:**
1. Opens the HDF5 file with `h5py.File`.
2. Reads the `X` dataset (all images) and the `Y` dataset (all labels).
3. Reads `class_names` from the file's attributes, stored as a JSON string.
4. Reads `img_shape` from attributes.
5. Returns a `Dataset`.

The entire array is loaded into RAM in one operation — this is why HDF5 has very fast load times.

---

#### Function: `load_npz(path)`

```python
def load_npz(path: str) -> Dataset
```

Loads a dataset from a `.npz` file saved by `save_npz`.

**Process:**
1. Calls `np.load(path, allow_pickle=True)`.
2. Extracts arrays `X`, `Y`, and `class_names`.
3. Returns a `Dataset`.

---

#### Function: `load_tfrecord(path)`

```python
def load_tfrecord(path: str) -> Dataset
```

Loads a dataset from a `.tfrecord` file saved by `save_tfrecord`. Also reads the accompanying `.meta.json` sidecar file for class names and image shape.

**Process:**
1. Reads `path + ".meta.json"` to get `class_names` and `img_shape`.
2. Opens the TFRecord file with `tf.data.TFRecordDataset`.
3. Parses each record using a parse function that decodes the raw image bytes back into a pixel array and reshapes it using the stored height/width/channels.
4. Batches records in groups of 512 for efficient reading.
5. Concatenates all batches into a single NumPy array X.
6. Returns a `Dataset`.

---

#### Function: `save_imagefolder(ds, root)`

```python
def save_imagefolder(ds: Dataset, root: str) -> None
```

Saves a `Dataset` back to the ImageFolder format — one subdirectory per class, one PNG file per image.

**Process:**
1. Creates class directories under `root/class_name/`.
2. For each image and its label, determines the class name from `ds.class_names[label]`.
3. Saves the image as a PNG file named `000000.png`, `000001.png`, etc. (six-digit zero-padded index).
4. For grayscale images (C=1), squeezes the channel dimension before saving so PIL can write a proper grayscale PNG.

---

#### Function: `save_hdf5(ds, path)`

```python
def save_hdf5(ds: Dataset, path: str) -> None
```

Saves a `Dataset` to a `.h5` file.

**Process:**
1. Creates an HDF5 file with two datasets: `X` (images) and `Y` (labels).
2. Stores `class_names` and `img_shape` as JSON strings in the file's attributes section (the HDF5 equivalent of file metadata).

---

#### Function: `save_npz(ds, path)`

```python
def save_npz(ds: Dataset, path: str) -> None
```

Saves a `Dataset` to a `.npz` file using `np.savez` (uncompressed). Stores arrays `X`, `Y`, and `class_names`.

---

#### Function: `save_tfrecord(ds, path)`

```python
def save_tfrecord(ds: Dataset, path: str) -> None
```

Saves a `Dataset` to a `.tfrecord` file and writes a sidecar `.meta.json`.

**Process:**
1. Opens a `tf.io.TFRecordWriter`.
2. For each image, creates a `tf.train.Example` protocol buffer containing:
   - `image`: the raw pixel bytes of the image.
   - `label`: the integer class index.
   - `height`, `width`, `channels`: the image dimensions (needed at load time to reshape the bytes back into an array).
3. Writes a `.meta.json` sidecar file with `class_names`, `img_shape`, and `num_samples` (the total count, needed for streaming).

**Why store height/width/channels per record?** TFRecord has no header — it is a flat sequence of records. Without storing dimensions in each record, it would be impossible to reshape the bytes back into an image during loading.

---

#### Function: `load(path, fmt, **kwargs)`

```python
def load(path: str, fmt=None, **kwargs) -> Dataset
```

The main public loading function. Dispatches to the correct format-specific loader.

- `fmt` is auto-detected if not provided.
- `**kwargs` are passed through to the underlying loader (e.g. `max_classes` for imagefolder).

---

#### Function: `save(ds, path, fmt)`

```python
def save(ds: Dataset, path: str, fmt=None) -> None
```

The main public saving function. Dispatches to the correct format-specific saver.

---

#### Function: `convert(src_path, dst_path, src_fmt, dst_fmt, **kwargs)`

```python
def convert(src_path, dst_path, src_fmt=None, dst_fmt=None, **kwargs) -> Dataset
```

Convenience function that combines load and save. Loads from `src_path` and immediately saves to `dst_path`. Returns the intermediate `Dataset` object.

**Example:**
```python
convert("my_dataset/", "my_dataset.h5")       # ImageFolder → HDF5
convert("my_dataset.h5", "my_dataset.npz")    # HDF5 → NPZ
convert("my_dataset.tfrecord", "output/")     # TFRecord → ImageFolder
```

---

#### Class: `ValidationResult`

```python
@dataclass
class ValidationResult:
    passed: bool
    n_samples_match: bool
    shape_match: bool
    dtype_match: bool
    value_range_match: bool
    label_mismatches: int
    label_distribution_match: bool
    avg_pixel_mse: float
    max_pixel_mse: float
    messages: list
```

Holds the outcome of a validation comparison between two datasets. Each field records whether a specific check passed or failed. The `messages` list contains human-readable lines that are printed by `print_report()`.

---

#### Function: `validate(ds1, ds2, num_samples)`

```python
def validate(ds1: Dataset, ds2: Dataset, num_samples: int = 20) -> ValidationResult
```

Compares two datasets and checks whether a conversion preserved the data perfectly. Returns a `ValidationResult`.

**Checks performed:**

| Check | What it verifies |
|---|---|
| Sample count | Both datasets have the same number of images |
| Image shape | Both datasets have images of the same dimensions (H, W, C) |
| Dtype | Both datasets use the same data type (uint8) |
| Value range | Both datasets have the same minimum and maximum pixel values |
| Labels | Every label in ds1 matches the corresponding label in ds2 |
| Class distribution | The number of images per class is identical in both datasets |
| Pixel MSE | For a random subset of images, the mean squared error between corresponding pixels is near zero |

The pixel MSE check is the most rigorous: it selects `num_samples` random images from both datasets and computes the average squared difference between their pixel values. For lossless formats (HDF5, NPZ, TFRecord with raw bytes), this should be exactly 0. The threshold is 1e-6.

---

### 6.2 `streaming_pipeline.py`

**Purpose**: Provides streaming (batch-by-batch) data access for all four formats via `tf.data.Dataset` objects. Unlike `universal_pipeline`, which loads the entire dataset into RAM, this module delivers data one batch at a time during training.

**Dependencies**: `numpy`, `tensorflow`, `h5py` (lazy), `Pillow` (lazy), `json`, `pathlib`

---

#### Class: `ThroughputCallback`

A Keras callback that records how long each training epoch takes, and from those times computes training throughput in samples per second.

**How Keras callbacks work**: Keras calls certain methods automatically during training. `on_epoch_begin` is called at the start of each epoch and `on_epoch_end` at the end. By recording the time at both events, the duration of each epoch is captured.

**Attributes:**
- `epoch_times`: a list of floats, one per epoch, recording wall-clock seconds.

**Method: `throughput(samples_per_epoch)`**
Computes total samples processed divided by total elapsed time across all epochs. Returns samples per second.

---

#### Class: `DatasetInfo`

```python
@dataclass(frozen=True)
class DatasetInfo:
    img_shape:   tuple
    num_classes: int
    num_samples: int   # -1 if unknown
    class_names: tuple
```

Lightweight metadata container. Unlike `Dataset`, it holds no pixel data — only descriptive information about the dataset. Used by the streaming functions to know the image shape and sample count without loading images.

`frozen=True` means the object is immutable after creation (its fields cannot be changed).

---

#### Function: `get_dataset_info(path, fmt, *, max_classes)`

```python
def get_dataset_info(path, fmt=None, *, max_classes=None) -> DatasetInfo
```

Reads metadata about a dataset without loading any pixel data.

- For **ImageFolder**: scans the directory tree to count files and opens only the first image to determine shape.
- For **HDF5**: opens the file and reads the array shape from the file header (no pixel data transferred).
- For **NPZ**: uses `mmap_mode='r'` (memory-mapped mode) to read the array shape from the file header without loading the array.
- For **TFRecord**: reads the `.meta.json` sidecar file.

---

#### Function: `stream(path, fmt, *, batch_size, shuffle, shuffle_buffer, max_classes)`

```python
def stream(path, fmt=None, *, batch_size=32, shuffle=True,
           shuffle_buffer=1000, max_classes=None) -> tf.data.Dataset
```

The main public function. Returns a `tf.data.Dataset` that yields tuples of `(images, labels)` where images are float32 in [0, 1] and labels are int64.

One full pass over the returned dataset equals one epoch. To train for multiple epochs, call `.repeat()` on the result (done automatically in `benchmark.py`).

**Parameters:**
- `batch_size`: number of images per batch delivered to the model.
- `shuffle`: whether to randomise the order of images each epoch.
- `shuffle_buffer`: for TFRecord only — the number of records held in memory for random sampling. Should be set to the full dataset size for a proper shuffle.
- `max_classes`: imagefolder only — restricts which class folders are loaded.

**Format-specific implementations:**

**TFRecord** (`_stream_tfrecord`): uses `tf.data.TFRecordDataset` natively with `num_parallel_reads=AUTOTUNE` (TensorFlow automatically decides how many threads to use), followed by a parallel `.map()` to parse records, optional `.shuffle()`, `.batch()`, and `.prefetch()`. The static H, W, C values from the meta.json are used in the reshape operation so that Keras can infer the output tensor shape for model building.

**HDF5** (`_stream_hdf5`): uses `tf.data.Dataset.from_generator()` wrapping a Python generator. The generator opens the HDF5 file once, shuffles an index array, and reads batches of rows using fancy indexing. h5py requires that fancy (non-sequential) index arrays be sorted, which is done with `np.sort()` before each read.

**NPZ** (`_stream_npz`): uses `from_generator()` with `np.load(path, mmap_mode='r')`. Memory-mapped mode means numpy does not load the entire array into RAM — instead, the operating system loads only the pages of the file that are actually accessed. This is the key mechanism that keeps RAM usage low.

**ImageFolder** (`_stream_imagefolder`): collects all `(file_path, label)` pairs from the directory tree upfront (this is just a directory scan, no image reading), then yields batches of opened images from a Python generator. Channel count is locked from the first readable image, matching the behaviour of `load_imagefolder`.

All streamers finish with `.prefetch(tf.data.AUTOTUNE)`, which tells TensorFlow to prepare the next batch in the background while the model processes the current one — this overlaps I/O and computation.

---

### 6.3 `benchmark.py`

**Purpose**: Orchestrates all experiments. Converts datasets, validates conversions, trains models, measures performance, and prints results tables.

**Dependencies**: `universal_pipeline`, `streaming_pipeline`, `MNIST_neural_network_training`, `TINYIMAGENET_model`, `numpy`, `time`, `tracemalloc`, `math`

---

#### Function: `_peak_ram_mb(fn, *args, **kwargs)`

Measures peak RAM usage of a single function call using Python's `tracemalloc` module.

`tracemalloc` instruments Python's memory allocator to track every allocation and de-allocation. `get_traced_memory()` returns the current and peak memory usage since tracking started. The peak is converted from bytes to megabytes.

**Important limitation**: `tracemalloc` only tracks Python-level allocations. Memory allocated directly by TensorFlow (on the C/CUDA side) is not counted. This means RAM figures for TFRecord (which involves TensorFlow) may appear lower than actual usage.

---

#### Function: `_size_mb(*paths)`

Recursively computes the total size on disk of all given paths, returning the result in megabytes. Handles both individual files and directories.

---

#### Function: `_run_pipeline(src_train, src_val, src_fmt, out_dir, load_kwargs)`

Shared steps 1–3 for any benchmark:
1. **Load** the source dataset from disk.
2. **Convert** it to all other three formats, saving them in `out_dir`.
3. **Validate** each conversion by loading the converted file and calling `validate()`.

Returns: the loaded source datasets, a dictionary of conversion times, and a dictionary of file paths for all four formats.

This function is called by all three benchmark functions to avoid code duplication.

---

#### Function: `_train_mnist(ds_train, ds_val, iterations, alpha)`

Trains the custom NumPy neural network on a given dataset and returns validation accuracy.

**Process:**
1. Reshapes X from (N, 28, 28, 1) to (784, N) — the format the MNIST neural network expects (pixels as rows, samples as columns).
2. Normalises pixel values to [0, 1] by dividing by 255.
3. Shuffles the training set.
4. Calls `gradient_descent()` to train.
5. Runs `forward_prop()` on the validation set and computes accuracy.

---

#### Function: `run_mnist_benchmark(...)`

Full benchmark for MNIST:
1. Runs `_run_pipeline()` (convert + validate all formats).
2. For each of the four formats:
   - Loads train and validation data (measuring RAM with `_peak_ram_mb`).
   - Measures file size on disk.
   - Trains and evaluates the MNIST neural network.
3. Prints a results table.

**Measured metrics:**
- `Convert(s)`: time to convert the source format to this format.
- `Load(s)`: time to load from disk into a `Dataset`.
- `RAM(MB)`: peak Python heap during loading.
- `File(MB)`: size of the dataset files on disk.
- `Train(s)`: time to complete all training iterations.
- `Accuracy`: fraction of validation samples correctly classified.

---

#### Function: `_train_tinyimagenet(ds_train, ds_val, epochs, batch_size)`

Trains the Keras CNN on a given dataset and returns validation accuracy.

**Process:**
1. Normalises both datasets to [0, 1] using `.normalized()`.
2. Creates a fresh Keras model with `create_model(num_classes)`.
3. Calls `model.fit()` with `verbose=1` so training progress is visible.
4. Evaluates the model on the validation set.

---

#### Function: `run_tinyimagenet_benchmark(...)`

Full benchmark for Tiny ImageNet — same structure as the MNIST benchmark but using the Keras CNN and configurable number of classes and epochs.

---

#### Function: `run_tinyimagenet_streaming_benchmark(...)`

Streaming benchmark for Tiny ImageNet. Instead of loading the full dataset into RAM, this function uses `stream()` from `streaming_pipeline.py`.

**Key differences from the full-load benchmark:**

- Calls `stream()` instead of `load()` to build `tf.data.Dataset` pipelines.
- Measures **first-batch latency**: the time from calling `stream()` until the first batch is actually delivered to the model. This is the real data-access startup cost — the moment the format first touches the disk.
- Applies `.repeat()` to the dataset so it cycles indefinitely, and passes `steps_per_epoch` to Keras to define epoch boundaries.
- Uses `ThroughputCallback` to record training throughput in samples per second.
- Sets `shuffle_buffer = ds_train_src.num_samples` to ensure TFRecord gets a full-quality shuffle (the entire dataset in the shuffle pool), comparable to the index-array shuffle used by other formats.

**Measured metrics:**
- `Convert(s)`: same as full-load.
- `1stBatch(s)`: wall-clock time until the model receives its first batch.
- `RAM(MB)`: peak Python heap during training (not loading, since there is no separate load phase).
- `File(MB)`: same as full-load.
- `Train(s)`: total training time.
- `Samples/s`: training throughput — total samples processed divided by total training time.
- `Accuracy`: validation accuracy at the last epoch.

---

#### Helper functions

**`_set_seed(seed)`**: Fixes the random state of Python's `random` module, NumPy, and TensorFlow in one call. Called before every training run so that weight initialisation is reproducible. When multiple runs are requested (`--runs N`), the seed is incremented by the run index (`seed + run_idx`) so each run is different but deterministic.

**`_auto_results_file(mode, dataset, max_classes)`**: Generates the output JSON filename automatically from the experiment parameters — e.g. `results_streaming_tinyimagenet_50classes.json` or `results_fullload_mnist.json`. This prevents results from different experiments mixing into the same file.

**`_append_results(filepath, ...)`**: Appends one entry to the JSON results file. If the file does not exist it is created; if it exists, the new entry is added to the existing list. Each entry records the mode, dataset, format, run index, seed, timestamp, and all measured metrics. Because multiple format runs (done in separate invocations for thermal isolation) all append to the same file, the JSON accumulates the full experiment gradually.

---

#### Command-Line Interface

`--mode` and `--dataset` are always required — there is no implicit default behaviour.

```
python benchmark.py --mode {fullload,streaming} --dataset {mnist,tinyimagenet} [options]

--mode        fullload | streaming     Required. fullload loads all data into RAM; streaming feeds data batch-by-batch.
--dataset     mnist | tinyimagenet     Required. Which dataset to benchmark.
--format      imagefolder | hdf5 | npz | tfrecord   Optional. Run only this format (omit to run all four).
--runs N                               Repetitions per format (default 1). Each run uses seed+run_idx.
--seed N                               Base random seed (default 42).
--max-classes N                        Tiny ImageNet: number of classes (default 5).
--epochs N                             Tiny ImageNet: training epochs (default 10).
--iterations N                         MNIST: training iterations (default 500).
--alpha F                              MNIST: learning rate (default 0.1).
```

**Usage examples:**
```bash
# Quick single-run checks:
python benchmark.py --mode fullload  --dataset mnist
python benchmark.py --mode fullload  --dataset tinyimagenet --max-classes 50
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50

# Thermally isolated, multi-run (recommended for reliable results):
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format imagefolder --runs 3
# (wait for MacBook to cool)
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format hdf5 --runs 3
# (wait...)
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format npz --runs 3
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format tfrecord --runs 3

# Aggregate all saved runs into one table:
python aggregate_results.py results_streaming_tinyimagenet_50classes.json
```

**Why `--mode` and `--dataset` are always required**: the previous design used implicit defaults (no `--streaming` flag meant full-load; the dataset was inferred). This made the interface ambiguous, especially when adding `--format` and `--runs`. Making both flags explicit ensures every invocation unambiguously describes the experiment being run.

---

### 6.4 `aggregate_results.py`

**Purpose**: Reads the JSON results file written by `benchmark.py` and produces a summary table with statistics across multiple runs. This is the final step of a multi-run, thermally isolated experiment.

**Dependencies**: `json`, `sys`, `numpy`

---

#### Why this file exists

When formats are run separately (one per invocation) and each is repeated 3 times, the raw results exist as 12 individual entries in a JSON file. The benchmark script prints per-run output to the terminal, but does not aggregate across runs. `aggregate_results.py` performs that aggregation and presents the final comparison in a clean table.

---

#### How it works

1. Loads the JSON file from disk.
2. Groups all entries by format name.
3. For each format, computes:
   - **Median** for timing metrics (load time, train time, first-batch latency, samples/s). Median is used instead of mean because a single thermal throttle spike — like an epoch that took 68s instead of 27s — would distort a mean significantly, while the median remains unaffected.
   - **Mean** for stable metrics (RAM, file size) that do not vary meaningfully between runs.
   - **Mean ± standard deviation** for accuracy, since this directly communicates both the expected value and the run-to-run variance.
4. Prints a formatted table.

---

#### Usage

```bash
python aggregate_results.py results_streaming_tinyimagenet_50classes.json
python aggregate_results.py results_fullload_mnist.json
```

**Example output (streaming, 3 runs per format):**
```
────────────────────────────────────────────────────────────────────────────────
  Format       Runs  Convert(s)  1stBatch(s)  RAM(MB)  File(MB)  Train(s)  Samples/s  Accuracy
────────────────────────────────────────────────────────────────────────────────
  imagefolder  3     —           0.017        27.4     376.6     289.000   901        0.7102 ± 0.0123
  hdf5         3     0.132       0.019        14.4     586.1     282.000   888        0.6934 ± 0.0098
  npz          3     0.431       0.116        1752.5   586.1     1078.000  232        0.6180 ± 0.0089
  tfrecord     3     0.716       0.315        3.5      591.0     275.000   907        0.7215 ± 0.0201
────────────────────────────────────────────────────────────────────────────────
  Timing: median across runs  |  RAM/File: mean across runs
  Accuracy: mean ± std across runs
```

The `±` value for accuracy is the key output: if it is small (e.g. ±0.01), format differences in accuracy are real. If it is large (e.g. ±0.05), the observed differences are within noise and not attributable to the format.

---

### 6.5 `MNIST_neural_network_training.py`

**Purpose**: Implements a two-layer neural network for MNIST from scratch using only NumPy. No deep learning framework is used. Also contains format-specific loaders for the original (pre-universal) MNIST pipeline and the `NeuralNetworkMNIST` class.

This file is part of the **original pipeline** but its core functions (`gradient_descent`, `forward_prop`, `get_predictions`, `get_accuracy`) are reused by `benchmark.py`.

---

#### Neural Network Architecture

The network has:
- **Input layer**: 784 neurons (one per pixel of a flattened 28×28 image).
- **Hidden layer**: 10 neurons with ReLU activation.
- **Output layer**: 10 neurons with softmax activation (one per digit class).

Total parameters: 10×784 + 10 + 10×10 + 10 = 7,960 numbers.

---

#### Core Mathematical Functions

**`init_params()`**: Initialises all four weight matrices/vectors (w1, b1, w2, b2) with small random values drawn from a uniform distribution centred at 0.

**`ReLU(Z)`**: The Rectified Linear Unit activation function. Returns `max(0, Z)` element-wise. Values below zero become zero; positive values pass through unchanged. This introduces non-linearity, allowing the network to learn complex patterns.

**`softmax(Z)`**: Converts a vector of raw scores into probabilities that sum to 1. Used in the output layer so the network's output can be interpreted as a confidence distribution over 10 digit classes. Numerically stabilised by subtracting the maximum value before exponentiation (prevents overflow).

**`forward_prop(w1, b1, w2, b2, X)`**: Computes the network's output for a batch of inputs X. Returns intermediate values z1, a1, z2, a2 needed for backpropagation.

```
X (784, N)  →  z1 = w1·X + b1  →  a1 = ReLU(z1)  →  z2 = w2·a1 + b2  →  a2 = softmax(z2)
```

**`one_hot(Y)`**: Converts integer labels to one-hot vectors. For MNIST, label 3 becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. Used to compute the error during backpropagation.

**`deriv_ReLU(Z)`**: The derivative of ReLU: returns 1 where Z > 0, else 0. Used in backpropagation.

**`back_prop(z1, a1, z2, a2, w2, Y, X)`**: Computes the gradients of the loss with respect to all weights and biases using the chain rule (backpropagation). Returns dW1, db1, dW2, db2.

**`update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)`**: Subtracts a fraction (`alpha`) of each gradient from the corresponding parameter. This is gradient descent — moving the parameters in the direction that reduces the loss.

**`get_predictions(a2)`**: Takes the output probabilities and returns the class with the highest probability for each sample using `argmax`.

**`get_accuracy(predictions, Y)`**: Computes the fraction of correct predictions.

**`gradient_descent(X, Y, iterations, alpha)`**: Runs the complete training loop for a fixed number of iterations: forward pass → backpropagation → parameter update. Prints accuracy every 10 iterations.

---

#### Class: `NeuralNetworkMNIST`

Object-oriented wrapper around the above functions, used by the original `MNIST_main.py`. Not used by `benchmark.py`.

Methods: `load_data()`, `preprocess_data()`, `train(iterations, alpha)`, `evaluate_on_test_set()`.

---

### 6.6 `MNIST_dataset_converter.py`

**Purpose**: Original MNIST-specific dataset container, converter, and validator. Part of the legacy pipeline, not used by `benchmark.py`.

---

#### Class: `MNISTDataset`

Simple container holding `X` (images as flat 784-element arrays) and `Y` (labels). Has a `normalize()` method that divides X by 255 in-place.

---

#### Class: `DatasetConverter`

Static methods for saving a `MNISTDataset` to each binary format. Does not save class names or image shape metadata — a limitation of the original design.

- `to_hdf5(dataset, path)`: writes X and Y datasets to an HDF5 file.
- `to_npz(dataset, path)`: writes X and Y using `np.savez`.
- `to_tfrecord(dataset, path)`: serialises each image as a `tf.train.Example`. Stores only `image` and `label` fields (no shape metadata).

---

#### Class: `DatasetValidator`

Compares two `MNISTDataset` objects.

- `compare_datasets(ds1, ds2, num_samples, visualize)`: checks sample count, labels, and pixel MSE on a random subset. Optionally shows side-by-side visualisations.
- `visualize_comparison(img1, img2, idx)`: displays three panels — original image, converted image, and pixel difference heatmap — using Matplotlib.

---

### 6.7 `MNIST_main.py`

**Purpose**: Original entry point for the MNIST pipeline. Part of the legacy pipeline.

**What it does:**
1. Loads MNIST from `mnist_imagefolder/train` and `mnist_imagefolder/test` using `load_from_imagefolder`.
2. Converts to HDF5, NPZ, TFRecord using `DatasetConverter`.
3. Validates each conversion using `DatasetValidator`.
4. Trains `NeuralNetworkMNIST` on each format.
5. Visualises predictions with `visualize_test_predictions`.

This script is self-contained and can be run independently of the universal pipeline.

---

### 6.8 `TINYIMAGENET_loader.py`

**Purpose**: Original Tiny ImageNet-specific loaders. Part of the legacy pipeline.

---

#### Function: `load_tinyimagenet_subset(root, split, max_classes)`

Loads a subset of Tiny ImageNet from its original directory structure.

The Tiny ImageNet directory layout is not a simple ImageFolder — it uses a specific structure:
```
tiny-imagenet-200/
  train/
    n01443537/       ← class directory (WordNet ID)
      images/
        image_001.JPEG
        ...
  val/
    images/          ← all validation images in one flat folder
      val_0.JPEG
      ...
    val_annotations.txt  ← maps filename → class
```

For the training split, images are read directly from each class's `images/` subdirectory.
For the validation split, a `val_annotations.txt` file is parsed to determine the class of each image. Only images belonging to the selected classes are loaded.

All images are opened with PIL, converted to RGB, and resized to 64×64.

---

#### Loaders: `load_from_hdf5`, `load_from_npz`, `load_from_tfrecord`

Format-specific loaders for the original Tiny ImageNet pipeline. Not used by `benchmark.py` (which uses `universal_pipeline.load()` instead).

---

### 6.9 `TINYIMAGENET_dataset_module.py`

**Purpose**: Original Tiny ImageNet dataset container, converter, and validator. Part of the legacy pipeline.

---

#### Class: `ImageDataset`

Container holding `X` (images as (N, 64, 64, 3) array), `Y` (labels), and `img_shape` (64, 64, 3). Has a `normalize()` method.

---

#### Class: `DatasetConverter`

Saves an `ImageDataset` to binary formats. The TFRecord saver includes `height`, `width`, and `channels` fields per record — an improvement over the MNIST version, which omitted shape metadata.

---

#### Class: `DatasetValidator`

Same structure as the MNIST validator, adapted to handle multi-channel colour images. `visualize_comparison` uses `imshow` without a greyscale colourmap since images are RGB.

---

### 6.10 `TINYIMAGENET_model.py`

**Purpose**: Defines the Keras CNN used for Tiny ImageNet classification.

---

#### Function: `create_model(num_classes)`

Creates and compiles a Keras Sequential CNN. Called fresh for each format experiment to ensure fair comparison (each format starts with the same random weight initialisation — though the randomness is not seeded, so values differ between runs).

**Architecture:**

```
Input: (64, 64, 3)
   ↓
Conv2D(32 filters, 3×3, ReLU)    → (62, 62, 32)
MaxPooling2D(2×2)                → (31, 31, 32)
   ↓
Conv2D(64 filters, 3×3, ReLU)    → (29, 29, 64)
MaxPooling2D(2×2)                → (14, 14, 64)
   ↓
Conv2D(64 filters, 3×3, ReLU)    → (12, 12, 64)
   ↓
Flatten                          → (9216,)
Dense(64, ReLU)                  → (64,)
Dense(num_classes, softmax)      → (num_classes,)
```

**Layers explained:**
- **Conv2D**: a convolutional layer that slides small filters (3×3 pixel windows) across the image to detect local features (edges, textures, shapes). The number of filters determines how many different features are detected.
- **MaxPooling2D**: reduces spatial dimensions by taking the maximum value in each 2×2 window. Reduces computation and makes the representation spatially invariant.
- **Flatten**: converts the 3D feature map into a 1D vector so it can be fed into Dense layers.
- **Dense**: a fully connected layer where every neuron connects to every input. The final Dense layer with softmax produces a probability distribution over classes.

**Compiled with:**
- Optimizer: Adam (adaptive learning rate — adjusts step size per parameter).
- Loss: sparse categorical cross-entropy (standard for multi-class classification with integer labels).
- Metric: accuracy.

---

### 6.11 `TINYIMAGENET_main.py`

**Purpose**: Original entry point for the Tiny ImageNet pipeline. Part of the legacy pipeline.

**What it does:**
1. Loads a subset of Tiny ImageNet using `load_tinyimagenet_subset`.
2. Converts to HDF5, NPZ, TFRecord using `DatasetConverter`.
3. Validates each conversion.
4. Trains the Keras CNN on each format, measuring load time and accuracy.
5. Prints a summary table.

Configurable via the `MAX_CLASSES` variable at the top of the file (default: 5).

---

## 7. Data Flow Diagrams

### 7.1 Full-Load Benchmark Flow

```
Raw images on disk
(mnist_imagefolder/ or tiny-imagenet-200/)
        │
        ▼
   universal_pipeline.load()
        │
        ▼
   Dataset object (all images in RAM as NumPy array)
        │
        ├──► save() → train.h5       ──► validate() ──► load() → Dataset
        ├──► save() → train.npz      ──► validate() ──► load() → Dataset
        ├──► save() → train.tfrecord ──► validate() ──► load() → Dataset
        │
        ▼ (for each format)
   dataset.normalized() → float32 arrays [0,1]
        │
        ▼
   model.fit(X_train, Y_train)  ←── MNIST NN or Keras CNN
        │
        ▼
   accuracy, train_time, load_time, RAM, file_size
        │
        ▼
   Results table
```

### 7.2 Streaming Benchmark Flow

```
Converted files on disk (from step above)
        │
        ▼
   streaming_pipeline.stream()  ← builds tf.data pipeline (no reading yet)
        │
        ▼
   tf.data.Dataset.repeat()     ← makes dataset infinite
        │
        ▼
   model.fit(stream, steps_per_epoch=N, epochs=E)
        │
        │  for each step:
        │    ┌── pipeline reads 1 batch from disk
        │    ├── pipeline decodes and normalises batch
        │    └── model processes batch (forward + backward pass)
        │
        ▼
   accuracy, throughput (samples/s), 1st-batch latency, RAM
        │
        ▼
   Results table
```

---

## 8. How to Run the Experiments

### Prerequisites

```bash
# Activate the Python environment (must be done before any command)
source tfg_environment/bin/activate
```

The following data must be present in the project directory:
- `mnist_imagefolder/train/` and `mnist_imagefolder/test/` — MNIST images in ImageFolder format.
- `tiny-imagenet-200/` — Tiny ImageNet dataset in its original structure.

### Quick single-run commands

These run all four formats in sequence and print a results table immediately. Useful for a quick check or small experiments.

```bash
# MNIST full-load benchmark (all 4 formats, 500 iterations)
python benchmark.py --mode fullload --dataset mnist

# Tiny ImageNet full-load benchmark (5 classes, 10 epochs)
python benchmark.py --mode fullload --dataset tinyimagenet --max-classes 5

# Tiny ImageNet full-load benchmark (50 classes)
python benchmark.py --mode fullload --dataset tinyimagenet --max-classes 50

# Tiny ImageNet streaming benchmark (5 classes)
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 5

# Tiny ImageNet streaming benchmark (50 classes)
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50
```

### Thermally isolated, multi-run commands (recommended)

Run one format per invocation, with time to cool the machine between runs. Results accumulate in a JSON file and are summarised at the end with `aggregate_results.py`.

```bash
# Example: streaming benchmark, 50 classes, 3 runs per format
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format imagefolder --runs 3
# (wait 5–10 minutes for MacBook to cool)
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format hdf5 --runs 3
# (wait...)
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format npz --runs 3
# (wait...)
python benchmark.py --mode streaming --dataset tinyimagenet --max-classes 50 --format tfrecord --runs 3

# Show the aggregated summary table
python aggregate_results.py results_streaming_tinyimagenet_50classes.json
```

The same pattern works for full-load and for any `--max-classes` value. The JSON filename is generated automatically from the flags, so results never mix between different experiments.

### Output

Each benchmark invocation prints:
1. A loading summary (number of samples, image shape, class count).
2. Conversion times per format.
3. Validation reports for each converted format.
4. Per-run training output (epoch-by-epoch) with seed displayed.
5. A results table (when running all formats in one go).
6. A line confirming the JSON file was saved.

`aggregate_results.py` prints a single summary table with median timing and mean ± std accuracy across all saved runs.

---

## 9. Understanding the Benchmark Results

### 9.1 What the Metrics Mean

| Metric | Unit | What it measures |
|---|---|---|
| Convert(s) | seconds | Time to write the dataset in this format. The source format shows `—` (no conversion needed). |
| Load(s) | seconds | Time to read the entire dataset from disk into RAM. |
| RAM(MB) | megabytes | Peak Python heap memory during loading. |
| File(MB) | megabytes | Total size of the dataset files on disk. |
| Train(s) | seconds | Total model training time after the data is in RAM. |
| Accuracy | fraction | Fraction of validation images correctly classified (1.0 = 100%). |
| 1stBatch(s) | seconds | (Streaming only) Time until the first batch is delivered to the model. |
| Samples/s | samples/second | (Streaming only) Average training throughput across all epochs. |

### 9.2 Expected Observations

**Load time**: ImageFolder is the slowest loader by a large margin for large datasets. For 50 classes (~25,000 images), ImageFolder takes ~6 seconds while HDF5 and NPZ take ~0.2 seconds — a 30× difference. This is because ImageFolder must open thousands of individual files, while HDF5/NPZ perform a single sequential binary read.

**RAM**: HDF5 and NPZ use roughly half the RAM of ImageFolder and TFRecord for loading. ImageFolder loads images one by one and builds a Python list before stacking — this temporarily holds two copies of the data. TFRecord uses TensorFlow's C++ runtime which allocates memory outside Python's tracker.

**File size**: ImageFolder (PNG files) is smaller than binary formats for small images like MNIST (PNG compression is effective on simple grayscale patterns). For colour images like Tiny ImageNet, binary formats are similar in size or smaller.

**Training time (full-load)**: After loading, all formats train at nearly the same speed because the model sees identical NumPy arrays. Observed differences are due to thermal throttling (the CPU heats up during long sequential runs) rather than the format itself.

**Accuracy**: Should be similar across formats for the same dataset, since all formats preserve the data exactly (validated by pixel MSE = 0). Observed differences are due to random weight initialisation — the model starts from different random parameters each time.

**Streaming throughput**: TFRecord achieves the highest throughput because it is natively supported by TensorFlow's multi-threaded I/O pipeline. ImageFolder is slower because it reads individual files through a Python generator. NPZ is the slowest because memory-mapped random access has higher overhead per batch than sequential reads.

### 9.3 Why Training Time Is Not a Meaningful Comparison (Full-Load)

In the full-load benchmark, training time differences between formats are not meaningful for two reasons:

1. **All formats train on identical NumPy arrays.** The format only affects the loading step. Once data is in RAM, the model sees the same numbers regardless of where they came from.
2. **Thermal throttling.** Running four ~5-minute experiments back to back on a laptop causes the CPU to heat up and reduce its clock speed progressively. The fourth format (TFRecord) always runs slower than the first (ImageFolder), not because TFRecord is inherently slower to train on, but because the machine is hotter by that point.

To obtain reliable training time comparisons, each format should be run in isolation with adequate cooling time between experiments.

---

## 10. Limitations and Methodological Notes

### 10.1 Small Dataset Bias

With only 5 classes (~2,500 Tiny ImageNet images), the entire dataset fits in the operating system's file cache after the first read. Subsequent format experiments effectively read from RAM rather than disk, making all formats appear equally fast. Meaningful I/O differences only appear at 50+ classes.

### 10.2 RAM Measurement Incompleteness

`tracemalloc` only tracks Python-level memory allocations. TensorFlow allocates significant memory directly in C++ (for graph execution, tensor buffers, and XLA compilation), which is invisible to `tracemalloc`. This means the RAM column underestimates actual memory usage for TFRecord and all streaming experiments.

### 10.3 Accuracy Variance and Reproducibility

Accuracy values from a single training run have high variance, especially on small datasets with few epochs. The random weight initialisation means the same format can achieve 0.62 in one run and 0.75 in another — a 13 percentage point difference that has nothing to do with the format.

The benchmark now addresses this in two ways:

- **Fixed random seeds**: `--seed` ensures each run uses a known seed, making individual runs reproducible. Multiple runs use `seed + run_idx` (e.g. 42, 43, 44) so they differ from each other but remain deterministic.
- **Multi-run statistics**: `--runs 3` (or more) repeats the training for a format and saves each result to JSON. `aggregate_results.py` then computes mean ± std, which makes variance visible. If the std is large (e.g. ±0.05), format differences smaller than that threshold are not meaningful.

To draw defensible conclusions about format-accuracy differences, 3 runs per format is considered acceptable for a thesis-level study.

### 10.4 Streaming vs. Full-Load Comparison

The streaming benchmark measures different things than the full-load benchmark. In streaming:
- Training time includes data loading (interleaved with computation).
- RAM stays low throughout training.
- The model never has access to the full dataset at once, which can slightly affect shuffle quality and training dynamics.

These are not flaws — they reflect a more realistic training scenario for large datasets that do not fit in RAM.

### 10.5 CPU-Only Training

All experiments run on CPU (Apple Silicon via macOS). On GPU systems, the I/O bottleneck would be more pronounced because the GPU processes batches much faster than the CPU can load them from disk. Format differences in streaming throughput would be amplified significantly on GPU hardware.
