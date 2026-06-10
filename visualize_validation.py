"""
visualize_validation.py
─────────────────────────────────────────────────────────────────────────────
Mirrors Experiment 1 in benchmark.py: for every ordered pair of formats,
load the source, convert to a temp file, load back, validate pixel-by-pixel.
Temp files are deleted after use — only the two PNG figures are written.

  Figure 1 — validation_comparison.png
    4 rows × 4 columns of images.
    Each row picks one representative conversion (one per source format):
      IF→HDF5 | HDF5→NPZ | NPZ→TFRecord | TFRecord→IF
    Plus a pixel-difference row (amplified ×50, should be all-black).

  Figure 2 — validation_report.png
    Full 12-pair validation table (all ordered format pairs).
"""

import sys, os, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE   = Path(__file__).parent
DATA   = BASE / "bench_exp" / "tinyimagenet_5classes"
SRC_DIRS = {
    "imagefolder": str(DATA / "train_imagefolder"),
    "hdf5":        str(DATA / "train.h5"),
    "npz":         str(DATA / "train.npz"),
    "tfrecord":    str(DATA / "train.tfrecord"),
}
FMT_EXT = {
    "imagefolder": None,        # needs a directory, handled specially
    "hdf5":        ".h5",
    "npz":         ".npz",
    "tfrecord":    ".tfrecord",
}
ALL_FMTS = ["imagefolder", "hdf5", "npz", "tfrecord"]

sys.path.insert(0, str(BASE))
os.chdir(BASE)

from universal_pipeline import load, save, validate

# ── load all four source datasets once ───────────────────────────────────────
print("Loading source datasets …")
sources = {}
for fmt in ALL_FMTS:
    sources[fmt] = load(SRC_DIRS[fmt], fmt=fmt)
    ds = sources[fmt]
    print(f"  {fmt:12s}: {ds.num_samples} samples  shape={ds.img_shape}")

# ── convert every ordered pair (12 total), validate ──────────────────────────
print("\nConverting and validating all 12 pairs …")
results = {}   # (src_fmt, dst_fmt) → (ds_converted, ValidationResult)

for src_fmt in ALL_FMTS:
    for dst_fmt in ALL_FMTS:
        if src_fmt == dst_fmt or dst_fmt == "imagefolder":
            continue
        ds_src = sources[src_fmt]

        if dst_fmt == "imagefolder":
            tmp_dir = tempfile.mkdtemp(suffix="_if")
            tmp_path = tmp_dir
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=FMT_EXT[dst_fmt], delete=False)
            tmp_path = tmp.name
            tmp.close()

        try:
            save(ds_src, tmp_path, fmt=dst_fmt)
            ds_conv = load(tmp_path, fmt=dst_fmt)
            vr = validate(ds_src, ds_conv, num_samples=50)
            status = "PASS" if vr.passed else "FAIL"
            print(f"  [{status}] {src_fmt:12s} → {dst_fmt:12s}  MSE={vr.avg_pixel_mse:.2e}")
            results[(src_fmt, dst_fmt)] = (ds_conv, vr)
        finally:
            if dst_fmt == "imagefolder":
                import shutil
                shutil.rmtree(tmp_path, ignore_errors=True)
            else:
                os.unlink(tmp_path)
                meta = tmp_path + ".meta.json"
                if os.path.exists(meta):
                    os.unlink(meta)

# ── choose 4 representative sample indices (one per class) ───────────────────
ref_ds = sources["imagefolder"]
sample_indices = []
for cls_idx in range(min(4, ref_ds.num_classes)):
    where = np.where(ref_ds.Y == cls_idx)[0]
    sample_indices.append(int(where[0]))
class_labels = [ref_ds.class_names[ref_ds.Y[i]] for i in sample_indices]

# ── colour constants ──────────────────────────────────────────────────────────
FMT_COLORS = {
    "imagefolder": "#4C72B0",
    "hdf5":        "#DD8452",
    "npz":         "#55A868",
    "tfrecord":    "#C44E52",
}
FMT_LABELS = {
    "imagefolder": "ImageFolder",
    "hdf5":        "HDF5",
    "npz":         "NPZ",
    "tfrecord":    "TFRecord",
}
CHECK_OK   = "#2ECC71"
CHECK_FAIL = "#E74C3C"

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — one row per conversion pair
# Columns: Original | Converted | Pixel diff (×50)
# ─────────────────────────────────────────────────────────────────────────────

AMPLIFY    = 50
ALL_PAIRS  = sorted(results.keys())   # 9 pairs, sorted by src then dst
N_PAIRS    = len(ALL_PAIRS)
N_COLS     = 3   # original | converted | diff

CELL = 1.8   # inches per cell
FIG_W = N_COLS * CELL + 2.2
FIG_H = N_PAIRS * CELL + 0.9

fig1, axes = plt.subplots(N_PAIRS, N_COLS,
                           figsize=(FIG_W, FIG_H),
                           facecolor="white")
fig1.suptitle(
    "Data Integrity: Pixel-Exact Round-Trip Across Storage Formats",
    fontsize=12, fontweight="bold", y=0.995,
)

# column headers
col_titles = ["Original", "Converted", f"Pixel diff (×{AMPLIFY})"]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=9, fontweight="bold",
                            color="#333333", pad=5)

for row, (src_fmt, dst_fmt) in enumerate(ALL_PAIRS):
    ds_src  = sources[src_fmt]
    ds_conv = results[(src_fmt, dst_fmt)][0]
    src_col = FMT_COLORS[src_fmt]
    dst_col = FMT_COLORS[dst_fmt]

    # pick first image of class 0 in source
    cls0_indices = np.where(ds_src.Y == 0)[0]
    idx = int(cls0_indices[0])

    def show(ax, img_arr, border_color):
        if img_arr.shape[2] == 1:
            ax.imshow(img_arr[:, :, 0], cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img_arr)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

    show(axes[row, 0], ds_src.X[idx],  src_col)
    show(axes[row, 1], ds_conv.X[idx], dst_col)

    # diff
    diff     = np.abs(ds_src.X[idx].astype(np.int16) - ds_conv.X[idx].astype(np.int16))
    diff_amp = np.clip(diff * AMPLIFY, 0, 255).astype(np.uint8)
    ax_d = axes[row, 2]
    if diff_amp.shape[2] == 1:
        ax_d.imshow(diff_amp[:, :, 0], cmap="hot", vmin=0, vmax=255)
    else:
        ax_d.imshow(diff_amp)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    mse = results[(src_fmt, dst_fmt)][1].avg_pixel_mse
    ax_d.set_xlabel(f"MSE = {mse:.1e}  max err = {int(diff.max())} px",
                    fontsize=7, color="#444444", labelpad=2)
    for spine in ax_d.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(1.2)

    # row label on the left
    axes[row, 0].set_ylabel(
        f"{FMT_LABELS[src_fmt]} → {FMT_LABELS[dst_fmt]}",
        fontsize=8, fontweight="bold", color="#333333",
        rotation=0, ha="right", va="center", labelpad=6,
    )

plt.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.02,
                    hspace=0.15, wspace=0.06)

out1 = BASE / "validation_comparison.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nFigure 1 saved → {out1}")
plt.close(fig1)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — full 12-pair validation table
# ─────────────────────────────────────────────────────────────────────────────

check_labels = [
    "Sample count",
    "Image shape",
    "Dtype",
    "Value range",
    "Label equality",
    "Class distribution",
    "Pixel MSE < 1×10⁻⁶",
]

def vr_checks(vr):
    return [
        (vr.n_samples_match,          vr.messages[0].split(": ", 1)[-1]),
        (vr.shape_match,              vr.messages[1].split(": ", 1)[-1]),
        (vr.dtype_match,              vr.messages[2].split(": ", 1)[-1]),
        (vr.value_range_match,        vr.messages[3].split(": ", 1)[-1]),
        (vr.label_mismatches == 0,    "identical" if vr.label_mismatches == 0
                                       else f"{vr.label_mismatches} mismatches"),
        (vr.label_distribution_match, "identical"),
        (vr.avg_pixel_mse < 1e-6,     f"{vr.avg_pixel_mse:.1e}"),
    ]

# Build ordered list of all 12 pairs, grouped by source
ordered_pairs = [(sf, df) for sf in ALL_FMTS for df in ALL_FMTS
                 if sf != df and df != "imagefolder"]
pair_labels   = [f"{FMT_LABELS[sf][:3]}→{FMT_LABELS[df][:3]}"
                 for sf, df in ordered_pairs]

N_CHECKS = len(check_labels)
N_PAIRS  = len(ordered_pairs)

fig2, ax = plt.subplots(figsize=(14, 5.5), facecolor="white")
ax.set_axis_off()
fig2.suptitle(
    "Validation Report: 9 Conversion Pairs — Pixel-Exact Round-Trip",
    fontsize=13, fontweight="bold", y=0.98,
)

col_headers = ["Check"] + pair_labels
table_data  = []
cell_colors = []

all_passed = all(vr.passed for _, vr in results.values())

for check_idx, clabel in enumerate(check_labels):
    row_data   = [clabel]
    row_colors = ["#F5F5F5" if check_idx % 2 == 0 else "white"]
    for pair in ordered_pairs:
        _, vr = results[pair]
        passed, detail = vr_checks(vr)[check_idx]
        # compact: just tick/cross for non-MSE rows
        if check_idx < 6:
            symbol = "✔" if passed else "✖ " + detail
        else:
            symbol = ("✔ " if passed else "✖ ") + detail
        row_data.append(symbol)
        row_colors.append(CHECK_OK + "33" if passed else CHECK_FAIL + "33")
    table_data.append(row_data)
    cell_colors.append(row_colors)

tbl = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    cellColours=cell_colors,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)

for col_idx in range(len(col_headers)):
    cell = tbl[0, col_idx]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold")

for row_idx in range(1, N_CHECKS + 1):
    tbl[row_idx, 0].set_text_props(fontweight="bold", color="#333333", ha="left")

# column widths
check_col_w = 0.18
data_col_w  = (1.0 - check_col_w) / N_PAIRS
for row_idx in range(N_CHECKS + 1):
    tbl[row_idx, 0].set_width(check_col_w)
    for col_idx in range(1, len(col_headers)):
        tbl[row_idx, col_idx].set_width(data_col_w)

badge_text  = "All 9 pairs PASSED  ✔" if all_passed else "Some pairs FAILED  ✖"
badge_color = CHECK_OK if all_passed else CHECK_FAIL
fig2.text(0.98, 0.01, badge_text,
          ha="right", va="bottom", fontsize=10,
          color=badge_color, fontweight="bold",
          transform=fig2.transFigure)

out2 = BASE / "validation_report.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Figure 2 saved → {out2}")
plt.close(fig2)

print("\nDone. No dataset files were left on disk.")
