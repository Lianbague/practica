"""
streaming_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Format-agnostic streaming data pipeline for image classification.

Unlike universal_pipeline.load(), which loads the entire dataset into RAM,
this module returns tf.data.Dataset objects that feed batches to Keras
one at a time — only the current batch is in memory during training.
This makes per-format RAM and throughput differences visible for large datasets.

Supported formats
  imagefolder  – root/class_name/**/*.jpg|png  (standard, nested OK)
  hdf5         – .h5  file  (X, Y datasets; class_names in attrs)
  npz          – .npz file  (X, Y, class_names arrays)
  tfrecord     – .tfrecord  + sidecar .meta.json

Public API
  get_dataset_info(path, fmt=None, *, max_classes=None) → DatasetInfo
  stream(path, fmt=None, *, batch_size=32, shuffle=True,
         shuffle_buffer=1000, max_classes=None)          → tf.data.Dataset
  ThroughputCallback  – Keras callback; records epoch_times and throughput()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
# ThroughputCallback — base class resolved at import time to avoid circular TF
# ─────────────────────────────────────────────────────────────────────────────

try:
    import tensorflow as _tf_eager
    _KERAS_BASE = _tf_eager.keras.callbacks.Callback
except Exception:
    _KERAS_BASE = object


class ThroughputCallback(_KERAS_BASE):
    """Records wall-clock time per epoch and computes aggregate throughput."""

    def __init__(self) -> None:
        if _KERAS_BASE is not object:
            super().__init__()
        self.epoch_times: list[float] = []
        self._t0: float = 0.0

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        self.epoch_times.append(time.perf_counter() - self._t0)

    def throughput(self, samples_per_epoch: int) -> float:
        """Return mean samples/second across all recorded epochs."""
        total = sum(self.epoch_times)
        return (samples_per_epoch * len(self.epoch_times)) / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DatasetInfo — lightweight metadata container, no pixel data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetInfo:
    """
    img_shape   : (H, W, C)
    num_classes : number of distinct classes
    num_samples : total sample count; -1 if unavailable (TFRecord without meta)
    class_names : ordered class labels
    """
    img_shape:   tuple
    num_classes: int
    num_samples: int
    class_names: tuple


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

_EXT_TO_FMT = {".h5": "hdf5", ".hdf5": "hdf5", ".npz": "npz", ".tfrecord": "tfrecord"}


def _detect_format(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        return "imagefolder"
    ext = p.suffix.lower()
    if ext in _EXT_TO_FMT:
        return _EXT_TO_FMT[ext]
    if not ext:
        return "imagefolder"
    raise ValueError(f"Cannot detect format from path: {path!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Metadata readers — open files but load no pixel data
# ─────────────────────────────────────────────────────────────────────────────

def _info_imagefolder(path: str, max_classes: Optional[int] = None) -> DatasetInfo:
    from PIL import Image as _PIL

    root = Path(path)
    class_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if max_classes is not None:
        class_dirs = class_dirs[:max_classes]

    class_names = tuple(d.name for d in class_dirs)
    num_samples = 0
    img_shape: Optional[tuple] = None

    for cls_dir in class_dirs:
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            num_samples += 1
            if img_shape is None:
                try:
                    img = _PIL.open(str(img_path))
                    if img.mode not in ("L", "RGB"):
                        img = img.convert("RGB")
                    arr = np.array(img)
                    img_shape = (arr.shape[0], arr.shape[1], 1 if arr.ndim == 2 else arr.shape[2])
                except Exception:
                    pass

    return DatasetInfo(
        img_shape=img_shape or (0, 0, 3),
        num_classes=len(class_names),
        num_samples=num_samples,
        class_names=class_names,
    )


def _info_hdf5(path: str) -> DatasetInfo:
    import h5py
    with h5py.File(path, "r") as f:
        img_shape   = tuple(f["X"].shape[1:])
        num_samples = int(f["X"].shape[0])
        class_names = tuple(json.loads(f.attrs.get("class_names", "[]")))
    return DatasetInfo(img_shape, len(class_names), num_samples, class_names)


def _info_npz(path: str) -> DatasetInfo:
    data        = np.load(path, mmap_mode="r", allow_pickle=True)
    img_shape   = tuple(data["X"].shape[1:])
    num_samples = int(data["X"].shape[0])
    class_names = tuple(data["class_names"].tolist()) if "class_names" in data else ()
    return DatasetInfo(img_shape, len(class_names), num_samples, class_names)


def _info_tfrecord(path: str) -> DatasetInfo:
    meta_path = path + ".meta.json"
    if not os.path.exists(meta_path):
        return DatasetInfo((0, 0, 3), 0, -1, ())
    with open(meta_path) as f:
        meta = json.load(f)
    class_names = tuple(meta.get("class_names", []))
    img_shape   = tuple(meta.get("img_shape", [0, 0, 3]))
    num_samples = int(meta.get("num_samples", -1))
    return DatasetInfo(img_shape, len(class_names), num_samples, class_names)


def get_dataset_info(
    path: str,
    fmt:  Optional[str] = None,
    *,
    max_classes: Optional[int] = None,
) -> DatasetInfo:
    """Read dataset metadata without loading any pixel data."""
    fmt = fmt or _detect_format(path)
    if fmt == "imagefolder":
        return _info_imagefolder(path, max_classes)
    if fmt == "hdf5":
        return _info_hdf5(path)
    if fmt == "npz":
        return _info_npz(path)
    if fmt == "tfrecord":
        return _info_tfrecord(path)
    raise ValueError(f"Unknown format: {fmt!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific streamers
# ─────────────────────────────────────────────────────────────────────────────

def _stream_tfrecord(
    path: str,
    batch_size: int,
    shuffle: bool,
    shuffle_buffer: int,
) -> "tf.data.Dataset":
    """
    TFRecord is the native tf.data format — full parallel reads and prefetch
    pipeline are used here, making this the fastest streaming option.
    Static H/W/C from meta.json are used in reshape so Keras sees a known
    output shape and can build the compute graph without error.
    """
    import tensorflow as tf

    meta_path = path + ".meta.json"
    H, W, C = 0, 0, 3
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        H, W, C = meta.get("img_shape", [0, 0, 3])

    def _parse(proto):
        feat = {
            "image":    tf.io.FixedLenFeature([], tf.string),
            "label":    tf.io.FixedLenFeature([], tf.int64),
            "height":   tf.io.FixedLenFeature([], tf.int64),
            "width":    tf.io.FixedLenFeature([], tf.int64),
            "channels": tf.io.FixedLenFeature([], tf.int64),
        }
        ex  = tf.io.parse_single_example(proto, feat)
        img = tf.io.decode_raw(ex["image"], tf.uint8)
        img = tf.reshape(img, (H, W, C))   # static shape → Flatten can infer output size
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(ex["label"], tf.int64)

    ds = tf.data.TFRecordDataset(path, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _stream_hdf5(
    path: str,
    batch_size: int,
    shuffle: bool,
) -> "tf.data.Dataset":
    import tensorflow as tf
    import h5py

    with h5py.File(path, "r") as f:
        n         = int(f["X"].shape[0])
        img_shape = tuple(f["X"].shape[1:])

    H, W, C = img_shape

    def _gen():
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        with h5py.File(path, "r") as f:
            for start in range(0, n, batch_size):
                # h5py fancy indexing requires sorted indices
                batch_idx = np.sort(indices[start : start + batch_size])
                yield (
                    f["X"][batch_idx].astype(np.float32) / 255.0,
                    f["Y"][batch_idx].astype(np.int64),
                )

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(None, H, W, C), dtype=tf.float32),
            tf.TensorSpec(shape=(None,),          dtype=tf.int64),
        ),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def _stream_npz(
    path: str,
    batch_size: int,
    shuffle: bool,
) -> "tf.data.Dataset":
    """
    mmap_mode='r' means numpy only loads the pages of the array that are
    actually accessed — the full array is never pulled into RAM at once.
    """
    import tensorflow as tf

    probe     = np.load(path, mmap_mode="r", allow_pickle=True)
    n         = int(probe["X"].shape[0])
    img_shape = tuple(probe["X"].shape[1:])
    del probe

    H, W, C = img_shape

    def _gen():
        data    = np.load(path, mmap_mode="r", allow_pickle=True)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield (
                data["X"][batch_idx].astype(np.float32) / 255.0,
                data["Y"][batch_idx].astype(np.int64),
            )

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(None, H, W, C), dtype=tf.float32),
            tf.TensorSpec(shape=(None,),          dtype=tf.int64),
        ),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def _stream_imagefolder(
    path: str,
    batch_size: int,
    shuffle: bool,
    max_classes: Optional[int] = None,
) -> "tf.data.Dataset":
    import tensorflow as tf
    from PIL import Image as _PIL

    root = Path(path)
    class_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if max_classes is not None:
        class_dirs = class_dirs[:max_classes]

    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

    # Collect (path, label) pairs — directory scan only, no I/O yet
    pairs: list[tuple[str, int]] = [
        (str(img_path), class_to_idx[cls_dir.name])
        for cls_dir in class_dirs
        for img_path in cls_dir.rglob("*")
        if img_path.suffix.lower() in _IMG_EXTS
    ]
    if not pairs:
        raise ValueError(f"No images found in {path!r}")

    # Lock spatial size and channel count from the first readable image
    _target_size:     Optional[tuple] = None
    _target_channels: Optional[int]   = None
    for p, _ in pairs:
        try:
            img = _PIL.open(p)
            if img.mode not in ("L", "RGB"):
                img = img.convert("RGB")
            arr = np.array(img)
            _target_size     = (arr.shape[0], arr.shape[1])
            _target_channels = 1 if arr.ndim == 2 else arr.shape[2]
            break
        except Exception:
            continue
    if _target_size is None:
        raise ValueError(f"Could not read any image from {path!r}")

    H, W, C = _target_size[0], _target_size[1], _target_channels

    def _open_img(p: str) -> np.ndarray:
        img = _PIL.open(p)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        if C == 3 and img.mode == "L":
            img = img.convert("RGB")
        elif C == 1 and img.mode == "RGB":
            img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr

    def _gen():
        order = list(range(len(pairs)))
        if shuffle:
            np.random.shuffle(order)
        for start in range(0, len(order), batch_size):
            imgs_list, labels_list = [], []
            for idx in order[start : start + batch_size]:
                p, label = pairs[idx]
                try:
                    imgs_list.append(_open_img(p))
                    labels_list.append(label)
                except Exception as exc:
                    print(f"  [warn] skipping {Path(p).name}: {exc}")
            if imgs_list:
                yield (
                    np.array(imgs_list, dtype=np.float32),
                    np.array(labels_list, dtype=np.int64),
                )

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(None, H, W, C), dtype=tf.float32),
            tf.TensorSpec(shape=(None,),          dtype=tf.int64),
        ),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def stream(
    path:           str,
    fmt:            Optional[str] = None,
    *,
    batch_size:     int  = 32,
    shuffle:        bool = True,
    shuffle_buffer: int  = 1000,
    max_classes:    Optional[int] = None,
) -> "tf.data.Dataset":
    """
    Return a tf.data.Dataset that streams the dataset batch-by-batch.

    One full pass over the returned dataset equals one epoch.  Keras handles
    multiple epochs automatically when ``epochs > 1`` is passed to
    ``model.fit()``.

    Parameters
    ----------
    path           : path to the dataset (file or directory)
    fmt            : format override; auto-detected from path when None
    batch_size     : number of samples per batch
    shuffle        : whether to randomise sample order each epoch
    shuffle_buffer : shuffle-buffer size for TFRecord (other formats shuffle
                     an index array, so this parameter is ignored for them)
    max_classes    : restrict to the first N class directories (imagefolder only)
    """
    fmt = fmt or _detect_format(path)
    if fmt == "tfrecord":
        return _stream_tfrecord(path, batch_size, shuffle, shuffle_buffer)
    if fmt == "hdf5":
        return _stream_hdf5(path, batch_size, shuffle)
    if fmt == "npz":
        return _stream_npz(path, batch_size, shuffle)
    if fmt == "imagefolder":
        return _stream_imagefolder(path, batch_size, shuffle, max_classes)
    raise ValueError(f"Unknown format: {fmt!r}")
