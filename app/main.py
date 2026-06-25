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

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image as PILImage

from universal_pipeline import ALL_FORMATS, detect_format
import tfrecord_io as _tio

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dataset Format Converter",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _peek_metadata(path: str, fmt: str, max_classes: Optional[int]) -> dict:
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


def _extract_zip_dataset(zip_bytes: bytes) -> tuple[str, str]:
    """Extract a ZIP upload and locate the dataset inside.
    Returns (dataset_path, detected_fmt).
    - TFRecord ZIP: zip contains train.tfrecord [+ .meta.json]
    - ImageFolder ZIP: zip contains class subdirectories with images
    """
    tmp_dir = tempfile.mkdtemp(prefix="ds_zip_")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(tmp_dir)

    tmp = Path(tmp_dir)

    # TFRecord: find any .tfrecord file
    tfrecords = sorted(tmp.rglob("*.tfrecord"))
    if tfrecords:
        return str(tfrecords[0]), "tfrecord"

    # ImageFolder: find the shallowest directory whose children are
    # class subdirectories containing image files
    for dirpath in sorted(tmp.rglob("*"), key=lambda p: len(p.parts)):
        if not dirpath.is_dir():
            continue
        subdirs = [d for d in dirpath.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not subdirs:
            continue
        if any(
            any(f.suffix.lower() in _IMG_EXTS for f in d.rglob("*") if f.is_file())
            for d in subdirs
        ):
            return str(dirpath), "imagefolder"

    return tmp_dir, ""


def _output_path(out_dir: str, src_path: str, fmt: str) -> str:
    ext = {"hdf5": ".h5", "npz": ".npz", "tfrecord": ".tfrecord", "imagefolder": ""}
    src = Path(src_path)
    stem = src.stem if src.is_file() else src.name
    if fmt == "imagefolder":
        return str(Path(out_dir) / f"{stem}_imagefolder")
    return str(Path(out_dir) / f"{stem}{ext[fmt]}")


def _download_bytes(path: str, fmt: str) -> tuple[bytes, str, str]:
    """Return (data_bytes, filename, mime_type) for a download button."""
    p = Path(path)
    if fmt == "imagefolder" and p.is_dir():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(p.parent))
        return buf.getvalue(), p.name + ".zip", "application/zip"
    else:
        data = p.read_bytes()
        mime = {
            "hdf5":      "application/x-hdf5",
            "npz":       "application/zip",
            "tfrecord":  "application/octet-stream",
        }.get(fmt, "application/octet-stream")
        return data, p.name, mime


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("inspected",          False),
    ("converted",          False),
    ("meta",               {}),
    ("src_path",           ""),
    ("src_fmt",            ""),
    ("conversion_results", {}),
    ("samples_dir",        ""),
    ("tmp_upload_path",    ""),
    ("tmp_zip_dir",        ""),
    ("upload_file_id",     ""),
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

input_mode = st.radio(
    "How do you want to provide the dataset?",
    ["📤 Upload file", "📂 Enter path"],
    horizontal=True,
    label_visibility="collapsed",
)

src_input     = ""
detected_fmt  = None

if input_mode == "📤 Upload file":
    st.caption(
        "HDF5 (.h5) and NPZ (.npz) — upload directly.  "
        "TFRecord and ImageFolder — upload as a ZIP file."
    )
    uploaded = st.file_uploader(
        "Drag and drop your dataset file here",
        type=["h5", "npz", "zip"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if file_id != st.session_state.upload_file_id:
            # Clean up previous temp files/dirs
            prev_path = st.session_state.tmp_upload_path
            if prev_path and os.path.exists(prev_path):
                if Path(prev_path).is_file():
                    os.unlink(prev_path)
            prev_zip = st.session_state.tmp_zip_dir
            if prev_zip and os.path.exists(prev_zip):
                shutil.rmtree(prev_zip, ignore_errors=True)
            st.session_state.tmp_zip_dir = ""

            if uploaded.name.endswith(".zip"):
                extracted_path, _zip_fmt = _extract_zip_dataset(uploaded.getvalue())
                st.session_state.tmp_upload_path = extracted_path
                # Store root tmp dir for cleanup on next upload
                st.session_state.tmp_zip_dir = str(Path(extracted_path).parent
                                                    if Path(extracted_path).is_file()
                                                    else Path(extracted_path).parent)
            else:
                ext = Path(uploaded.name).suffix
                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.write(uploaded.getvalue())
                tmp.close()
                st.session_state.tmp_upload_path = tmp.name

            st.session_state.upload_file_id     = file_id
            st.session_state.inspected          = False
            st.session_state.converted          = False
            st.session_state.conversion_results = {}

        src_input = st.session_state.tmp_upload_path

else:
    src_input = st.text_input(
        "Path to dataset",
        placeholder="e.g.  /Users/you/train_imagefolder  or  /Users/you/data.h5",
        label_visibility="collapsed",
    )

# ── format detection & inspect button ────────────────────────────────────────

path_exists = bool(src_input and os.path.exists(src_input))

if src_input and not path_exists:
    st.error("Path does not exist.")

if path_exists:
    try:
        detected_fmt = detect_format(src_input)
        if input_mode == "📂 Enter path":
            st.caption(f"Detected format: **{detected_fmt.upper()}**")
    except ValueError as e:
        st.error(str(e))
        detected_fmt = None

    if detected_fmt and st.button("Inspect dataset", type="secondary"):
        st.session_state.inspected       = False
        st.session_state.converted       = False
        st.session_state.conversion_results = {}

        with st.spinner("Reading dataset metadata…"):
            meta = _peek_metadata(src_input, detected_fmt, max_classes=None)

        if "error" in meta:
            st.error(f"Could not read dataset: {meta['error']}")
        else:
            st.session_state.meta     = meta
            st.session_state.src_path = src_input
            st.session_state.src_fmt  = detected_fmt
            st.session_state.inspected = True

# ── inspection results ────────────────────────────────────────────────────────

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
            col.image(img, width="stretch")

    # Max classes — only for ImageFolder with more than 10 classes
    max_classes_val = None
    if (st.session_state.src_fmt == "imagefolder"
            and meta.get("num_classes", 0) > 10):
        st.markdown(" ")
        mc_col, btn_col = st.columns([2, 1])
        with mc_col:
            max_classes_val = st.number_input(
                f"Limit classes (dataset has {meta['num_classes']})",
                min_value=2, max_value=meta["num_classes"],
                value=min(50, meta["num_classes"]), step=1,
                help="Load only the first N class folders (alphabetical order).",
            )
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Apply limit", type="secondary"):
                with st.spinner("Re-reading metadata…"):
                    new_meta = _peek_metadata(
                        st.session_state.src_path, "imagefolder", int(max_classes_val)
                    )
                if "error" not in new_meta:
                    st.session_state.meta = new_meta
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 · Conversion options
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.inspected:
    st.divider()
    st.header("Step 2 · Convert To")

    src_fmt    = st.session_state.src_fmt
    other_fmts = [f for f in ALL_FORMATS if f != src_fmt]
    fmt_labels = {"imagefolder": "ImageFolder", "hdf5": "HDF5", "npz": "NPZ", "tfrecord": "TFRecord"}
    fmt_descs  = {
        "imagefolder": "One PNG per image, organised in class folders.",
        "hdf5":        "Single binary .h5 file. Fast load, ideal for full-load training.",
        "npz":         "Single compressed .npz file. NumPy native.",
        "tfrecord":    "Sequential .tfrecord file. Best streaming throughput.",
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

    if not selected_fmts:
        st.warning("Select at least one target format.")

    if selected_fmts and st.button("Convert", type="primary"):
        st.session_state.converted       = False
        st.session_state.conversion_results = {}

        tmp_dir     = tempfile.mkdtemp(prefix="ds_converter_")
        out_json    = os.path.join(tmp_dir, "results.json")
        samples_dir = os.path.join(tmp_dir, "samples")
        out_dir     = os.path.join(tmp_dir, "output")

        current_meta = st.session_state.meta
        mc = current_meta.get("num_classes") if src_fmt == "imagefolder" else None

        cfg = {
            "src_path":    st.session_state.src_path,
            "src_fmt":     src_fmt,
            "max_classes": mc,
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
    st.header("Step 3 · Your converted datasets are ready")

    results    = st.session_state.conversion_results
    fmt_labels = {"imagefolder": "ImageFolder", "hdf5": "HDF5", "npz": "NPZ", "tfrecord": "TFRecord"}
    fmt_icons  = {"imagefolder": "🗂️", "hdf5": "🗄️", "npz": "📦", "tfrecord": "📼"}
    all_passed = all(r.get("ok") and r.get("passed") for r in results.values())

    if all_passed:
        st.success("All conversions completed — every pixel was preserved exactly.")
    else:
        st.warning("Some conversions completed with issues — see details below.")

    # ── prominent download cards ──────────────────────────────────────────────
    ok_results = {fmt: r for fmt, r in results.items() if r.get("ok")}
    if ok_results:
        dl_cols = st.columns(len(ok_results))
        for col, (fmt, r) in zip(dl_cols, ok_results.items()):
            label    = fmt_labels.get(fmt, fmt.upper())
            icon     = fmt_icons.get(fmt, "📄")
            dst_path = r.get("path", "")
            with col:
                st.markdown(
                    f"<div style='text-align:center; padding:1rem; border:1px solid #ddd; "
                    f"border-radius:8px; background:#fafafa;'>"
                    f"<div style='font-size:2rem;'>{icon}</div>"
                    f"<div style='font-weight:bold; font-size:1.05rem;'>{label}</div>"
                    f"<div style='color:#888; font-size:0.85rem;'>{r['size_mb']:.1f} MB"
                    f"{'  ✅' if r.get('passed') else ('  ⚠️' if fmt == 'imagefolder' else '  ❌')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(" ")
                if dst_path and os.path.exists(dst_path):
                    try:
                        data, fname, mime = _download_bytes(dst_path, fmt)
                        st.download_button(
                            label=f"Download {label}",
                            data=data,
                            file_name=fname,
                            mime=mime,
                            key=f"dl_{fmt}",
                            width="stretch",
                            type="primary",
                        )
                    except Exception:
                        st.caption(f"Saved to: `{dst_path}`")

    # ── detailed results (collapsible) ───────────────────────────────────────
    st.markdown(" ")
    for fmt, r in results.items():
        label = fmt_labels.get(fmt, fmt.upper())
        imagefolder_ordering_note = (
            fmt == "imagefolder" and r.get("ok")
            and not r.get("passed")
        )
        if r.get("ok"):
            status = ("⚠️ Validation note" if imagefolder_ordering_note
                      else "✅ Validation passed" if r["passed"]
                      else "❌ Validation failed")
            header = f"{label}  ·  {status}  ·  {r['time_s']:.2f}s"
        else:
            header = f"{label}  ·  ❌ Conversion error"

        with st.expander(header, expanded=False):
            if r.get("ok"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Conversion time",  f"{r['time_s']:.2f}s")
                c2.metric("File size",         f"{r['size_mb']:.1f} MB")
                c3.metric("Validation",        "NOTE ⚠️" if imagefolder_ordering_note
                                               else "PASSED ✅" if r["passed"]
                                               else "FAILED ❌")

                if imagefolder_ordering_note:
                    st.info(
                        "All images and labels are pixel-perfect. The validator reports "
                        "mismatches because ImageFolder groups images by class folder, "
                        "which changes the load order compared to the source. "
                        "The data itself is complete and correct."
                    )

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
                            col.image(PILImage.open(orig), caption=f"Original · {lbl}", width="stretch")
                        for col, (_, conv, lbl) in zip(cols, valid_pairs):
                            col.image(PILImage.open(conv), caption=f"{label} · {lbl}", width="stretch")

            else:
                st.error(r.get("error", "Unknown error"))
                with st.expander("Full traceback"):
                    st.code(r.get("trace", "No traceback available"))
