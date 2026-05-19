"""
app/main.py  —  Dataset Format Converter
─────────────────────────────────────────────────────────────────────────────
Streamlit web app that converts image classification datasets between
ImageFolder, HDF5, NPZ, and TFRecord formats.

Run from the project root:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

# Make the project root importable (universal_pipeline, etc.)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image as PILImage

from universal_pipeline import ALL_FORMATS, detect_format
import tfrecord_io as _tio

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dataset Format Converter",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight helpers  (no full dataset load)
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _peek_metadata(path: str, fmt: str, max_classes: Optional[int]) -> dict:
    """Return basic metadata without loading the full dataset into RAM."""
    try:
        if fmt == "imagefolder":
            p = Path(path)
            class_dirs = sorted(d for d in p.iterdir() if d.is_dir() and not d.name.startswith("."))
            if max_classes:
                class_dirs = class_dirs[:max_classes]
            class_names = [d.name for d in class_dirs]
            n_samples = sum(
                1 for d in class_dirs
                for f in d.rglob("*") if f.suffix.lower() in _IMG_EXTS
            )
            img_shape = None
            for d in class_dirs:
                for f in d.rglob("*"):
                    if f.suffix.lower() in _IMG_EXTS:
                        img = PILImage.open(f)
                        w, h = img.size
                        c = 1 if img.mode == "L" else 3
                        img_shape = (h, w, c)
                        break
                if img_shape:
                    break

        elif fmt == "hdf5":
            import h5py
            with h5py.File(path, "r") as f:
                n_samples   = int(f["X"].shape[0])
                img_shape   = tuple(f["X"].shape[1:])
                class_names = json.loads(f.attrs.get("class_names", "[]"))

        elif fmt == "npz":
            data        = np.load(path, mmap_mode="r", allow_pickle=True)
            n_samples   = int(data["X"].shape[0])
            img_shape   = tuple(data["X"].shape[1:])
            class_names = data["class_names"].tolist() if "class_names" in data else []

        elif fmt == "tfrecord":
            meta_path = path + ".meta.json"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                n_samples   = meta.get("num_samples", -1)
                img_shape   = tuple(meta.get("img_shape", [0, 0, 3]))
                class_names = meta.get("class_names", [])
            else:
                n_samples, img_shape, class_names = -1, None, []

        else:
            return {}

        return {
            "num_samples": n_samples,
            "img_shape":   img_shape,
            "num_classes": len(class_names),
            "class_names": class_names,
        }

    except Exception as exc:
        return {"error": str(exc)}


def _peek_images(path: str, fmt: str, n: int = 6) -> list:
    """Return up to n sample images as display-ready uint8 arrays."""
    images = []
    try:
        if fmt == "imagefolder":
            p = Path(path)
            for cls_dir in sorted(p.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for img_file in cls_dir.rglob("*"):
                    if img_file.suffix.lower() in _IMG_EXTS:
                        img = np.array(PILImage.open(img_file).convert("RGB"))
                        images.append(img)
                        if len(images) >= n:
                            return images

        elif fmt == "hdf5":
            import h5py
            with h5py.File(path, "r") as f:
                chunk = f["X"][:n]
            for img in chunk:
                images.append(_to_display(img))

        elif fmt == "npz":
            data = np.load(path, mmap_mode="r", allow_pickle=True)
            for img in data["X"][:n]:
                images.append(_to_display(img))

        elif fmt == "tfrecord":
            count = 0
            for raw in _tio.read_tfrecord(path):
                img_bytes, _, h, w, c = _tio._decode_example(raw)
                arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, c).copy()
                images.append(_to_display(arr))
                count += 1
                if count >= n:
                    break

    except Exception:
        pass
    return images


def _to_display(arr: np.ndarray) -> np.ndarray:
    """Convert (H,W,1) grayscale to (H,W) so st.image renders correctly."""
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    return arr


def _file_size_mb(path: str) -> float:
    p = Path(path)
    total = 0
    if p.is_dir():
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    elif p.exists():
        total = p.stat().st_size
        meta = Path(path + ".meta.json")
        if meta.exists():
            total += meta.stat().st_size
    return total / 1024 ** 2


def _output_path(out_dir: str, src_path: str, fmt: str) -> str:
    ext = {"hdf5": ".h5", "npz": ".npz", "tfrecord": ".tfrecord", "imagefolder": ""}
    src = Path(src_path)
    stem = src.stem if src.is_file() else src.name
    if fmt == "imagefolder":
        return str(Path(out_dir) / f"{stem}_imagefolder")
    return str(Path(out_dir) / f"{stem}{ext[fmt]}")


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("inspected", False),
    ("converted", False),
    ("meta", {}),
    ("src_path", ""),
    ("src_fmt", ""),
    ("conversion_results", {}),
    ("samples_dir", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("Dataset Format Converter")
st.markdown(
    "Convert image classification datasets between "
    "**ImageFolder**, **HDF5**, **NPZ**, and **TFRecord** — "
    "with automatic validation that every pixel was preserved exactly."
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 · Source dataset
# ─────────────────────────────────────────────────────────────────────────────

st.header("Step 1 · Source Dataset")

col_path, col_mc = st.columns([4, 1])
with col_path:
    src_input = st.text_input(
        "Path to dataset",
        placeholder="e.g.  /Users/you/mnist_imagefolder/train  or  /Users/you/data.h5",
        label_visibility="collapsed",
    )
with col_mc:
    max_classes_input = st.number_input(
        "Max classes", min_value=1, max_value=1000, value=50, step=1,
        help="Only used when source is ImageFolder — limits how many class folders are loaded.",
    )

path_exists = src_input and os.path.exists(src_input)

if src_input and not path_exists:
    st.error("Path does not exist.")

if path_exists:
    try:
        detected_fmt = detect_format(src_input)
        st.caption(f"Detected format: **{detected_fmt.upper()}**")
    except ValueError as e:
        st.error(str(e))
        detected_fmt = None

    if detected_fmt:
        if st.button("Inspect dataset", type="secondary"):
            st.session_state.inspected = False
            st.session_state.converted = False
            st.session_state.conversion_results = {}

            with st.spinner("Reading dataset metadata…"):
                mc   = int(max_classes_input) if detected_fmt == "imagefolder" else None
                meta = _peek_metadata(src_input, detected_fmt, mc)

            if "error" in meta:
                st.error(f"Could not read dataset: {meta['error']}")
            else:
                st.session_state.meta      = meta
                st.session_state.src_path  = src_input
                st.session_state.src_fmt   = detected_fmt
                st.session_state.inspected = True

# ── Inspection results ───────────────────────────────────────────────────────

if st.session_state.inspected:
    meta = st.session_state.meta

    m1, m2, m3 = st.columns(3)
    m1.metric("Samples",     meta["num_samples"] if meta.get("num_samples", -1) != -1 else "unknown")
    m2.metric("Classes",     meta.get("num_classes", "—"))
    m3.metric("Image shape", str(meta.get("img_shape", "—")))

    if meta.get("class_names"):
        names       = meta["class_names"]
        preview_str = ", ".join(names[:20]) + (f"  … (+{len(names)-20} more)" if len(names) > 20 else "")
        st.caption(f"Classes: {preview_str}")

    with st.spinner("Loading sample images…"):
        sample_imgs = _peek_images(st.session_state.src_path, st.session_state.src_fmt, n=6)
    if sample_imgs:
        st.markdown("**Sample images:**")
        cols = st.columns(len(sample_imgs))
        for col, img in zip(cols, sample_imgs):
            col.image(img, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 · Conversion options
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.inspected:
    st.divider()
    st.header("Step 2 · Convert To")

    src_fmt      = st.session_state.src_fmt
    other_fmts   = [f for f in ALL_FORMATS if f != src_fmt]
    fmt_labels   = {"imagefolder": "ImageFolder", "hdf5": "HDF5", "npz": "NPZ", "tfrecord": "TFRecord"}
    fmt_descs    = {
        "imagefolder": "One PNG file per image, organised in class folders.",
        "hdf5":        "Single binary .h5 file. Fast load, good for full-load training.",
        "npz":         "Single compressed .npz file. NumPy native, supports memory mapping.",
        "tfrecord":    "Sequential .tfrecord file. TensorFlow native, best streaming throughput.",
    }

    st.markdown("Select the formats you want to produce:")
    fmt_cols = st.columns(len(other_fmts))
    selected_fmts = []
    for col, fmt in zip(fmt_cols, other_fmts):
        with col:
            checked = st.checkbox(fmt_labels[fmt], value=True, key=f"chk_{fmt}")
            st.caption(fmt_descs[fmt])
            if checked:
                selected_fmts.append(fmt)

    st.markdown(" ")
    default_out = str(Path(st.session_state.src_path).parent / "converted")
    out_dir = st.text_input("Output directory", value=default_out)

    convert_ready = bool(selected_fmts and out_dir)
    if not convert_ready:
        st.warning("Select at least one target format and provide an output directory.")

    if convert_ready and st.button("Convert", type="primary", use_container_width=False):
        st.session_state.converted = False
        st.session_state.conversion_results = {}

        tmp_dir     = tempfile.mkdtemp(prefix="ds_converter_")
        out_json    = os.path.join(tmp_dir, "results.json")
        samples_dir = os.path.join(tmp_dir, "samples")

        cfg = {
            "src_path":    st.session_state.src_path,
            "src_fmt":     src_fmt,
            "max_classes": int(max_classes_input) if src_fmt == "imagefolder" else None,
            "out_dir":     out_dir,
            "target_fmts": selected_fmts,
            "out_json":    out_json,
            "samples_dir": samples_dir,
        }

        worker     = Path(__file__).parent / "worker.py"
        stderr_log = os.path.join(tmp_dir, "worker_stderr.txt")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"]        = "1"
        env["TF_NUM_INTRAOP_THREADS"] = "1"
        env["TF_NUM_INTEROP_THREADS"] = "1"
        env["TF_CPP_MIN_LOG_LEVEL"]   = "3"
        env["PYTHONUNBUFFERED"]       = "1"

        with st.spinner("Converting… this may take a minute for large datasets."):
            with open(stderr_log, "w") as stderr_f:
                subprocess.run(
                    [sys.executable, str(worker), json.dumps(cfg)],
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_f,
                    cwd=str(_ROOT),
                    start_new_session=True,
                    env=env,
                )

        if not os.path.exists(out_json):
            st.error("Conversion worker crashed before producing results.")
            stderr_text = open(stderr_log).read() if os.path.exists(stderr_log) else ""
            if stderr_text:
                with st.expander("Error details"):
                    st.code(stderr_text[-3000:])
        else:
            with open(out_json) as f:
                raw = json.load(f)

            if "__load_error__" in raw:
                st.error(f"Failed to load source dataset: {raw['__load_error__']}")
            else:
                st.session_state.conversion_results = raw
                st.session_state.samples_dir        = samples_dir
                st.session_state.converted          = True
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 · Results
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.converted:
    st.divider()
    st.header("Step 3 · Results")

    results    = st.session_state.conversion_results
    fmt_labels = {"imagefolder": "ImageFolder", "hdf5": "HDF5", "npz": "NPZ", "tfrecord": "TFRecord"}
    all_passed = all(r.get("ok") and r.get("passed") for r in results.values())

    if all_passed:
        st.success("All conversions completed and validated successfully.")
    else:
        st.warning("Some conversions completed with issues — see details below.")

    for fmt, r in results.items():
        label = fmt_labels.get(fmt, fmt.upper())
        if r.get("ok"):
            status = "✅ Passed" if r["passed"] else "❌ Validation failed"
            header = f"{label}  ·  {status}  ·  {r['time_s']:.2f}s  ·  {r['size_mb']:.1f} MB"
        else:
            header = f"{label}  ·  ❌ Error"

        with st.expander(header, expanded=True):
            if r.get("ok"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Conversion time", f"{r['time_s']:.2f}s")
                c2.metric("File size on disk", f"{r['size_mb']:.1f} MB")
                c3.metric("Validation", "PASSED ✅" if r["passed"] else "FAILED ❌")

                st.markdown(f"**Saved to:** `{r['path']}`")

                st.markdown("**Validation checks:**")
                for msg in r.get("messages", []):
                    st.markdown(msg)

                n_s         = r.get("n_samples", 0)
                labels      = r.get("labels", [])
                samples_dir = st.session_state.get("samples_dir", "")

                if n_s and samples_dir:
                    orig_paths  = [os.path.join(samples_dir, f"orig_{i}.png") for i in range(n_s)]
                    conv_paths  = [os.path.join(samples_dir, f"{fmt}_{i}.png") for i in range(n_s)]
                    valid_pairs = [
                        (o, c, lbl)
                        for o, c, lbl in zip(orig_paths, conv_paths, labels)
                        if os.path.exists(o) and os.path.exists(c)
                    ]

                    if valid_pairs:
                        st.markdown("---")
                        st.markdown("**Visual comparison — original vs converted**")
                        cols = st.columns(len(valid_pairs))

                        for col, (orig, _, lbl) in zip(cols, valid_pairs):
                            col.image(PILImage.open(orig), caption=f"Original · {lbl}", use_container_width=True)

                        for col, (_, conv, lbl) in zip(cols, valid_pairs):
                            col.image(PILImage.open(conv), caption=f"{label} · {lbl}", use_container_width=True)

            else:
                st.error(r.get("error", "Unknown error"))
                with st.expander("Full traceback"):
                    st.code(r.get("trace", "No traceback available"))
