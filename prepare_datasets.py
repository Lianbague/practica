"""
prepare_datasets.py
─────────────────────────────────────────────────────────────────────────────
Downloads CIFAR-10 and Fashion MNIST via Keras, converts them to all four
storage formats, and validates every conversion pair (9 pairs per dataset).

Output layout (mirrors bench_exp/tinyimagenet_*classes/):
  bench_exp/cifar10/
    train_imagefolder/   val_imagefolder/
    train.h5             val.h5
    train.npz            val.npz
    train.tfrecord       val.tfrecord

  bench_exp/fashionmnist/
    (same structure)

Run from the project root with the virtual environment active:
    python prepare_datasets.py
"""

import os, sys, tempfile, types
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

# macOS ARM64 / pyarrow crash fix (must be before any TF import)
if "pyarrow" not in sys.modules:
    _pa = types.ModuleType("pyarrow")
    _pa.__version__ = "0.0.0"
    _pa.__getattr__ = lambda name: types.ModuleType(f"pyarrow.{name}")
    sys.modules.setdefault("pyarrow", _pa)
    for _sub in ("pyarrow.lib", "pyarrow.compute", "pyarrow.ipc",
                 "pyarrow.parquet", "pyarrow.csv", "pyarrow.dataset"):
        sys.modules.setdefault(_sub, types.ModuleType(_sub))

os.environ["KERAS_BACKEND"]               = "tensorflow"
os.environ["TF_DISABLE_PLUGGABLE_DEVICE"] = "1"

from universal_pipeline import Dataset, save, load, validate

# ── dataset definitions ───────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
FASHION_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot",
]

def load_cifar10() -> tuple[Dataset, Dataset]:
    from tensorflow.keras.datasets import cifar10
    (X_tr, Y_tr), (X_va, Y_va) = cifar10.load_data()
    # X already (N, 32, 32, 3) uint8; Y is (N, 1) → flatten
    return (
        Dataset(X=X_tr, Y=Y_tr.flatten().astype(np.int32),
                img_shape=(32, 32, 3), class_names=CIFAR10_CLASSES),
        Dataset(X=X_va, Y=Y_va.flatten().astype(np.int32),
                img_shape=(32, 32, 3), class_names=CIFAR10_CLASSES),
    )

def load_fashionmnist() -> tuple[Dataset, Dataset]:
    from tensorflow.keras.datasets import fashion_mnist
    (X_tr, Y_tr), (X_va, Y_va) = fashion_mnist.load_data()
    # X is (N, 28, 28) → add channel dim
    X_tr = X_tr[:, :, :, np.newaxis]
    X_va = X_va[:, :, :, np.newaxis]
    return (
        Dataset(X=X_tr, Y=Y_tr.astype(np.int32),
                img_shape=(28, 28, 1), class_names=FASHION_CLASSES),
        Dataset(X=X_va, Y=Y_va.astype(np.int32),
                img_shape=(28, 28, 1), class_names=FASHION_CLASSES),
    )

# ── helpers ───────────────────────────────────────────────────────────────────

ALL_FMTS = ["imagefolder", "hdf5", "npz", "tfrecord"]
FMT_EXT  = {"hdf5": ".h5", "npz": ".npz", "tfrecord": ".tfrecord"}

def fmt_path(out_dir: Path, split: str, fmt: str) -> str:
    if fmt == "imagefolder":
        return str(out_dir / f"{split}_imagefolder")
    return str(out_dir / f"{split}{FMT_EXT[fmt]}")

def save_all_formats(ds: Dataset, out_dir: Path, split: str) -> None:
    for fmt in ALL_FMTS:
        path = fmt_path(out_dir, split, fmt)
        if fmt == "imagefolder" and Path(path).exists():
            print(f"    [{fmt}] already exists, skipping")
            continue
        if fmt != "imagefolder" and Path(path).exists():
            print(f"    [{fmt}] already exists, skipping")
            continue
        print(f"    saving {fmt} …", end=" ", flush=True)
        save(ds, path, fmt=fmt)
        size_mb = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6 \
                  if Path(path).is_dir() else Path(path).stat().st_size / 1e6
        print(f"{size_mb:.1f} MB")

def validate_all_pairs(out_dir: Path, split: str) -> None:
    """Validate all 9 conversion pairs (no imagefolder as destination)."""
    print(f"\n  Validating {split} split …")
    all_pass = True
    for src_fmt in ALL_FMTS:
        for dst_fmt in ALL_FMTS:
            if src_fmt == dst_fmt or dst_fmt == "imagefolder":
                continue
            src_path = fmt_path(out_dir, split, src_fmt)
            dst_path = fmt_path(out_dir, split, dst_fmt)

            # convert src → tmp, load back, compare
            ext = FMT_EXT[dst_fmt]
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                ds_src  = load(src_path, fmt=src_fmt)
                save(ds_src, tmp_path, fmt=dst_fmt)
                ds_conv = load(tmp_path, fmt=dst_fmt)
                vr = validate(ds_src, ds_conv, num_samples=100)
                status = "PASS" if vr.passed else "FAIL"
                if not vr.passed:
                    all_pass = False
                print(f"    [{status}] {src_fmt:12s} → {dst_fmt:10s}  "
                      f"MSE={vr.avg_pixel_mse:.2e}  "
                      f"labels={'ok' if vr.label_mismatches==0 else str(vr.label_mismatches)+' mismatches'}")
            finally:
                os.unlink(tmp_path)
                meta = tmp_path + ".meta.json"
                if os.path.exists(meta):
                    os.unlink(meta)

    print(f"  → {'All pairs PASSED ✔' if all_pass else 'Some pairs FAILED ✖'}")

def prepare(name: str, loader_fn, out_dir: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Downloading / loading …")
    ds_train, ds_val = loader_fn()
    print(f"    train: {ds_train.num_samples} samples  shape={ds_train.img_shape}  "
          f"classes={ds_train.num_classes}")
    print(f"    val  : {ds_val.num_samples} samples  shape={ds_val.img_shape}")

    print("\n  Saving train split …")
    save_all_formats(ds_train, out_dir, "train")
    print("  Saving val split …")
    save_all_formats(ds_val,   out_dir, "val")

    validate_all_pairs(out_dir, "train")
    validate_all_pairs(out_dir, "val")

# ── main ──────────────────────────────────────────────────────────────────────

prepare("CIFAR-10",      load_cifar10,      Path("bench_exp/cifar10"))
prepare("Fashion MNIST", load_fashionmnist, Path("bench_exp/fashionmnist"))

print("\n\nDone. Datasets ready under bench_exp/cifar10/ and bench_exp/fashionmnist/")
