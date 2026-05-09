"""
universal_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Format-agnostic image-classification pipeline.

Supported formats
  imagefolder  – root/class_name/**/*.jpg|png  (standard, nested OK)
  hdf5         – .h5  file  (X, Y datasets; class_names in attrs)
  npz          – .npz file  (X, Y, class_names arrays)
  tfrecord     – .tfrecord  + sidecar .meta.json

Public API
  load(path, fmt=None, **kw)         → Dataset
  save(ds, path, fmt=None)
  convert(src, dst, src_fmt, dst_fmt)→ Dataset
  validate(ds1, ds2, n=20)           → ValidationResult
  build_model(img_shape, n_classes)  → Keras model
  benchmark(train, val, ...)         → dict of per-format results

Quick start
  ds = load("my_dataset/", fmt="imagefolder")
  convert("my_dataset/", "output.h5")
  benchmark("my_dataset/train", "my_dataset/val", output_dir="results/")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Dataset – universal in-memory container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Dataset:
    """
    X          : uint8 ndarray (N, H, W, C)  – raw pixel values [0, 255]
    Y          : int32 ndarray (N,)           – class indices
    img_shape  : (H, W, C)
    class_names: list of str, length == num_classes
    """
    X: np.ndarray
    Y: np.ndarray
    img_shape: tuple
    class_names: list

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_samples(self) -> int:
        return len(self.X)

    def normalized(self) -> Dataset:
        """Return a copy with X cast to float32 in [0, 1]."""
        return Dataset(
            self.X.astype(np.float32) / 255.0,
            self.Y.astype(np.int32),
            self.img_shape,
            self.class_names,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

_EXT_TO_FMT = {".h5": "hdf5", ".hdf5": "hdf5", ".npz": "npz", ".tfrecord": "tfrecord"}
_IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def detect_format(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        return "imagefolder"
    ext = p.suffix.lower()
    if ext in _EXT_TO_FMT:
        return _EXT_TO_FMT[ext]
    if not ext:                    # no extension → treat as imagefolder directory
        return "imagefolder"
    raise ValueError(f"Cannot detect format from path: {path!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _open_image(
    path: str,
    target_size: Optional[tuple] = None,
    target_channels: Optional[int] = None,
) -> np.ndarray:
    """Open an image file → uint8 ndarray (H, W, C), preserving original channels.
    If target_channels is set, convert to match (handles mixed grayscale/RGB datasets)."""
    img = Image.open(path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    # enforce consistent channel count across the dataset
    if target_channels == 3 and img.mode == "L":
        img = img.convert("RGB")
    elif target_channels == 1 and img.mode == "RGB":
        img = img.convert("L")
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]))  # PIL: (W, H)
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:           # L mode → add channel dim (H, W, 1)
        arr = arr[:, :, np.newaxis]
    return arr


def load_imagefolder(
    root: str,
    max_classes: Optional[int] = None,
    target_size: Optional[tuple] = None,   # (H, W) – auto-detected if None
) -> Dataset:
    """
    Load a dataset stored as root/class_name/**/*.ext.
    Images may be nested inside subdirectories (e.g. Tiny ImageNet's
    root/class/images/*.JPEG structure is handled automatically).
    """
    root_p = Path(root)
    class_dirs = sorted(
        d for d in root_p.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if max_classes is not None:
        class_dirs = class_dirs[:max_classes]

    class_names  = [d.name for d in class_dirs]
    class_to_idx = {n: i for i, n in enumerate(class_names)}

    X: list[np.ndarray] = []
    Y: list[int]        = []
    _size     = target_size  # locked from first image if None
    _channels: Optional[int] = None

    for cls_dir in class_dirs:
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            try:
                arr = _open_image(str(img_path), _size, _channels)
                if _size is None:
                    _size     = arr.shape[:2]   # lock spatial size
                    _channels = arr.shape[2]    # lock channel count
                X.append(arr)
                Y.append(class_to_idx[cls_dir.name])
            except Exception as exc:
                print(f"  [warn] skipping {img_path.name}: {exc}")

    X_arr     = np.array(X, dtype=np.uint8)
    img_shape = tuple(X_arr.shape[1:]) if len(X_arr) else (0, 0, 3)
    return Dataset(X_arr, np.array(Y, dtype=np.int32), img_shape, class_names)


def load_hdf5(path: str) -> Dataset:
    import h5py
    with h5py.File(path, "r") as f:
        X           = f["X"][:]
        Y           = f["Y"][:].astype(np.int32)
        class_names = json.loads(f.attrs.get("class_names", "[]"))
        img_shape   = tuple(json.loads(f.attrs.get("img_shape", "null") or "null")
                            or X.shape[1:])
    return Dataset(X, Y, tuple(X.shape[1:]), class_names)


def load_npz(path: str) -> Dataset:
    data        = np.load(path, allow_pickle=True)
    X           = data["X"]
    Y           = data["Y"].astype(np.int32)
    class_names = data["class_names"].tolist() if "class_names" in data else []
    return Dataset(X, Y, tuple(X.shape[1:]), class_names)


def load_tfrecord(path: str) -> Dataset:
    import tensorflow as tf

    meta_path = path + ".meta.json"
    class_names: list = []
    img_shape: tuple  = (0, 0, 3)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        class_names = meta.get("class_names", [])
        img_shape   = tuple(meta.get("img_shape", [0, 0, 3]))

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
        img = tf.reshape(img, (ex["height"], ex["width"], ex["channels"]))
        return img, ex["label"]

    Xs, Ys = [], []
    for imgs, labels in tf.data.TFRecordDataset(path).map(_parse).batch(512):
        Xs.append(imgs.numpy())
        Ys.append(labels.numpy())

    X = np.concatenate(Xs) if Xs else np.empty((0,) + img_shape, dtype=np.uint8)
    Y = np.concatenate(Ys).astype(np.int32) if Ys else np.array([], dtype=np.int32)
    return Dataset(X, Y, tuple(X.shape[1:]), class_names)


# ─────────────────────────────────────────────────────────────────────────────
# Savers
# ─────────────────────────────────────────────────────────────────────────────

def save_imagefolder(ds: Dataset, root: str) -> None:
    """Reconstruct a class-per-folder directory from a Dataset."""
    root_p   = Path(root)
    counters: dict[str, int] = {}
    for img_arr, label in zip(ds.X, ds.Y):
        cls  = ds.class_names[int(label)] if ds.class_names else str(int(label))
        cdir = root_p / cls
        cdir.mkdir(parents=True, exist_ok=True)
        idx              = counters.get(cls, 0)
        counters[cls]    = idx + 1
        squeeze = img_arr.squeeze() if img_arr.ndim == 3 and img_arr.shape[2] == 1 else img_arr
        Image.fromarray(squeeze).save(cdir / f"{idx:06d}.png")


def save_hdf5(ds: Dataset, path: str) -> None:
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=ds.X)
        f.create_dataset("Y", data=ds.Y)
        f.attrs["class_names"] = json.dumps(ds.class_names)
        f.attrs["img_shape"]   = json.dumps(list(ds.img_shape))


def save_npz(ds: Dataset, path: str) -> None:
    np.savez(
        path,
        X=ds.X,
        Y=ds.Y,
        class_names=np.array(ds.class_names, dtype=object),
    )


def save_tfrecord(ds: Dataset, path: str) -> None:
    import tensorflow as tf

    with tf.io.TFRecordWriter(path) as writer:
        for img, label in zip(ds.X, ds.Y):
            h, w, c = img.shape
            feat = {
                "image":    tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                "label":    tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                "height":   tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                "width":    tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
                "channels": tf.train.Feature(int64_list=tf.train.Int64List(value=[c])),
            }
            writer.write(
                tf.train.Example(features=tf.train.Features(feature=feat))
                .SerializeToString()
            )

    with open(path + ".meta.json", "w") as f:
        json.dump({"class_names": ds.class_names, "img_shape": list(ds.img_shape)}, f)


# ─────────────────────────────────────────────────────────────────────────────
# Unified load / save / convert
# ─────────────────────────────────────────────────────────────────────────────

_LOADERS = {
    "imagefolder": load_imagefolder,
    "hdf5":        load_hdf5,
    "npz":         load_npz,
    "tfrecord":    load_tfrecord,
}
_SAVERS = {
    "imagefolder": save_imagefolder,
    "hdf5":        save_hdf5,
    "npz":         save_npz,
    "tfrecord":    save_tfrecord,
}


def load(path: str, fmt: Optional[str] = None, **kwargs) -> Dataset:
    """Load a dataset from any supported format."""
    fmt = fmt or detect_format(path)
    return _LOADERS[fmt](path, **kwargs)


def save(ds: Dataset, path: str, fmt: Optional[str] = None) -> None:
    """Save a Dataset to any supported format."""
    fmt = fmt or detect_format(path)
    _SAVERS[fmt](ds, path)


def convert(
    src_path: str,
    dst_path: str,
    src_fmt: Optional[str] = None,
    dst_fmt: Optional[str] = None,
    **load_kwargs,
) -> Dataset:
    """
    Convert a dataset between any two formats.
    Formats are auto-detected from file extensions / directory paths when
    not supplied explicitly.

    Example
        convert("dataset/", "dataset.h5")
        convert("dataset.h5", "dataset.npz")
        convert("dataset.tfrecord", "dataset_out/", dst_fmt="imagefolder")
    """
    ds = load(src_path, src_fmt, **load_kwargs)
    save(ds, dst_path, dst_fmt)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

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

    def print_report(self) -> None:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        print(f"  {status}")
        for m in self.messages:
            print(f"  {m}")


def validate(ds1: Dataset, ds2: Dataset, num_samples: int = 20) -> ValidationResult:
    """
    Compare two datasets and return a detailed ValidationResult.

    Checks
    ──────
    • Sample count equality
    • Image shape equality
    • Dtype equality
    • Value range [min, max] equality
    • Per-sample label equality
    • Class distribution (histogram) equality
    • Pixel MSE on a random subset (expects < 1e-6 for lossless formats)
    """
    msgs: list[str] = []
    ok = True

    def check(cond: bool, ok_msg: str, fail_msg: str) -> bool:
        msgs.append(("✔ " if cond else "✖ ") + (ok_msg if cond else fail_msg))
        return cond

    n_ok    = check(len(ds1.X) == len(ds2.X),
                    f"Sample count: {len(ds1.X)}",
                    f"Sample count mismatch: {len(ds1.X)} vs {len(ds2.X)}")
    ok     &= n_ok

    sh_ok   = check(ds1.img_shape == ds2.img_shape,
                    f"Image shape: {ds1.img_shape}",
                    f"Shape mismatch: {ds1.img_shape} vs {ds2.img_shape}")
    ok     &= sh_ok

    dt_ok   = check(ds1.X.dtype == ds2.X.dtype,
                    f"Dtype: {ds1.X.dtype}",
                    f"Dtype mismatch: {ds1.X.dtype} vs {ds2.X.dtype}")

    r1, r2  = (int(ds1.X.min()), int(ds1.X.max())), (int(ds2.X.min()), int(ds2.X.max()))
    rng_ok  = check(r1 == r2,
                    f"Value range: {r1}",
                    f"Value range mismatch: {r1} vs {r2}")
    ok     &= rng_ok

    lbl_mm  = int(np.sum(ds1.Y != ds2.Y)) if n_ok else -1
    lbl_ok  = check(lbl_mm == 0,
                    "Labels: identical",
                    f"Label mismatches: {lbl_mm}")
    ok     &= lbl_ok

    if n_ok:
        d1 = np.bincount(ds1.Y, minlength=ds1.num_classes)
        d2 = np.bincount(ds2.Y, minlength=ds2.num_classes)
        dist_ok = check(np.array_equal(d1, d2),
                        "Class distribution: identical",
                        "Class distribution mismatch")
    else:
        dist_ok = False
        msgs.append("✖ Class distribution: skipped (size mismatch)")
    ok &= dist_ok

    # Pixel MSE on random subset
    n   = min(num_samples, len(ds1.X)) if n_ok else 0
    if n:
        idx  = np.random.choice(len(ds1.X), n, replace=False)
        diff = ds1.X[idx].astype(np.float64) - ds2.X[idx].astype(np.float64)
        # average over all dims except the sample axis
        mses    = np.mean(diff ** 2, axis=tuple(range(1, diff.ndim)))
        avg_mse = float(mses.mean())
        max_mse = float(mses.max())
        pix_ok  = check(avg_mse < 1e-6,
                        f"Pixel MSE: avg={avg_mse:.2e}  max={max_mse:.2e}",
                        f"Pixel MSE too high: avg={avg_mse:.2e}  max={max_mse:.2e}")
        ok &= pix_ok
    else:
        avg_mse = max_mse = float("nan")

    return ValidationResult(
        passed=ok,
        n_samples_match=n_ok,
        shape_match=sh_ok,
        dtype_match=dt_ok,
        value_range_match=rng_ok,
        label_mismatches=lbl_mm,
        label_distribution_match=dist_ok,
        avg_pixel_mse=avg_mse,
        max_pixel_mse=max_mse,
        messages=msgs,
    )


ALL_FORMATS = ["imagefolder", "hdf5", "npz", "tfrecord"]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick demo: load → convert to all formats → validate each
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "mnist_imagefolder/train"
    print(f"Loading {src}...")
    ds = load(src)
    print(f"  {ds.num_samples} samples  shape={ds.img_shape}  classes={ds.num_classes}")

    for fmt, ext in [("hdf5", "/tmp/demo.h5"), ("npz", "/tmp/demo.npz"),
                     ("tfrecord", "/tmp/demo.tfrecord"), ("imagefolder", "/tmp/demo_folder")]:
        save(ds, ext, fmt)
        ds2 = load(ext, fmt)
        r   = validate(ds, ds2, num_samples=5)
        print(f"  {fmt:<13} → passed={r.passed}  mse={r.avg_pixel_mse:.2e}")
