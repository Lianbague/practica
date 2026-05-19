"""
app/worker.py
─────────────────────────────────────────────────────────────────────────────
Subprocess worker launched by app/main.py. Runs load / save / validate in
isolation so that any C-extension crash cannot kill the Streamlit process.

Reads a JSON config from argv[1], writes results to config["out_json"].
Sample images are saved as PNGs under config["samples_dir"].
"""

import sys
from pathlib import Path

# Make project root importable (universal_pipeline, etc.)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import time
import traceback

import numpy as np
from PIL import Image as PILImage

from universal_pipeline import load, save, validate
import tfrecord_io as _tio   # pure-Python TFRecord — no TF needed


# ── helpers ───────────────────────────────────────────────────────────────────

def _file_size_mb(path: str) -> float:
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 2
    total = p.stat().st_size if p.exists() else 0
    meta  = Path(path + ".meta.json")
    if meta.exists():
        total += meta.stat().st_size
    return total / 1024 ** 2


def _output_path(out_dir: str, src_path: str, fmt: str) -> str:
    ext  = {"hdf5": ".h5", "npz": ".npz", "tfrecord": ".tfrecord", "imagefolder": ""}
    src  = Path(src_path)
    stem = src.stem if src.is_file() else src.name
    if fmt == "imagefolder":
        return str(Path(out_dir) / f"{stem}_imagefolder")
    return str(Path(out_dir) / f"{stem}{ext[fmt]}")


def _save_png(arr: np.ndarray, path: str) -> None:
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    PILImage.fromarray(arr).save(path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = json.loads(sys.argv[1])

    src_path    = cfg["src_path"]
    src_fmt     = cfg["src_fmt"]
    max_classes = cfg.get("max_classes")
    out_dir     = cfg["out_dir"]
    target_fmts = cfg["target_fmts"]
    out_json    = cfg["out_json"]
    samples_dir = cfg["samples_dir"]

    os.makedirs(out_dir,     exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    results = {}

    # ── load source ───────────────────────────────────────────────────────────
    try:
        load_kw = {"max_classes": max_classes} if src_fmt == "imagefolder" and max_classes else {}
        ds = load(src_path, src_fmt, **load_kw)
    except Exception as exc:
        results["__load_error__"] = str(exc)
        with open(out_json, "w") as f:
            json.dump(results, f)
        sys.exit(1)

    n_show  = min(4, ds.num_samples)
    indices = np.random.choice(ds.num_samples, n_show, replace=False)

    for pos, idx in enumerate(indices):
        _save_png(ds.X[idx], str(Path(samples_dir) / f"orig_{pos}.png"))

    # ── convert, validate, save sample PNGs ──────────────────────────────────
    for fmt in target_fmts:
        dst = _output_path(out_dir, src_path, fmt)
        try:
            t0 = time.time()
            if fmt == "tfrecord":
                _tio.save(ds, dst)
            else:
                save(ds, dst, fmt)
            elapsed = time.time() - t0

            ds2 = _tio.load(dst) if fmt == "tfrecord" else load(dst, fmt)
            vr  = validate(ds, ds2, num_samples=30)

            for pos, idx in enumerate(indices):
                _save_png(ds2.X[idx], str(Path(samples_dir) / f"{fmt}_{pos}.png"))

            label_list = [
                ds.class_names[int(ds.Y[idx])] if ds.class_names else str(int(ds.Y[idx]))
                for idx in indices
            ]

            results[fmt] = {
                "path":      dst,
                "time_s":    elapsed,
                "size_mb":   _file_size_mb(dst),
                "passed":    bool(vr.passed),
                "messages":  vr.messages,
                "labels":    label_list,
                "n_samples": n_show,
                "ok":        True,
            }

        except Exception as exc:
            results[fmt] = {
                "ok":    False,
                "error": str(exc),
                "trace": traceback.format_exc(),
            }

    with open(out_json, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
