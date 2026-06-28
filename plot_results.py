"""
plot_results.py — Generate benchmark result figures for the thesis article.
Run: python plot_results.py
Outputs PNG files in ./figures/
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── output directory ──────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
FORMAT_COLORS = {
    "imagefolder": "#4C72B0",
    "hdf5":        "#DD8452",
    "npz":         "#55A868",
    "tfrecord":    "#C44E52",
}
FORMAT_LABELS = {
    "imagefolder": "ImageFolder",
    "hdf5":        "HDF5",
    "npz":         "NPZ",
    "tfrecord":    "TFRecord",
}
FORMAT_MARKERS = {
    "imagefolder": "o",
    "hdf5":        "s",
    "npz":         "^",
    "tfrecord":    "D",
}
FORMATS = ["imagefolder", "hdf5", "npz", "tfrecord"]

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def mean_std(records, key):
    vals = [r[key] for r in records if key in r]
    return np.mean(vals), np.std(vals)

def get_streaming_stats(dataset, classes, metric="total_train_s"):
    path = f"results_exp3_streaming_{dataset}_{classes}classes.json"
    if not os.path.exists(path):
        return {}
    data = load_json(path)
    result = {}
    for fmt in FORMATS:
        recs = [r for r in data if r["fmt"] == fmt]
        if recs:
            m, s = mean_std(recs, metric)
            result[fmt] = (m, s)
    return result

def get_fullload_stats(dataset, classes, metric="total_train_s"):
    path = f"results_exp3_fullload_{dataset}_{classes}classes.json"
    if not os.path.exists(path):
        return {}
    data = load_json(path)
    result = {}
    for fmt in FORMATS:
        recs = [r for r in data if r["fmt"] == fmt]
        if recs:
            m, s = mean_std(recs, metric)
            result[fmt] = (m, s)
    return result

def get_exp2_stats(dataset, classes, metric="load_s"):
    path = f"results_exp2_{dataset}_{classes}classes.json"
    if not os.path.exists(path):
        return {}
    data = load_json(path)
    result = {}
    for fmt in FORMATS:
        recs = [r for r in data if r["fmt"] == fmt]
        if recs:
            m, s = mean_std(recs, metric)
            result[fmt] = (m, s)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Streaming: Total training time vs number of classes (Tiny ImageNet)
# ══════════════════════════════════════════════════════════════════════════════

TI_CLASSES = [5, 10, 25, 50, 100]

fig, ax = plt.subplots(figsize=(7, 4.5))

for fmt in FORMATS:
    xs, ys, errs = [], [], []
    for cl in TI_CLASSES:
        stats = get_streaming_stats("tinyimagenet", cl)
        if fmt in stats:
            m, s = stats[fmt]
            xs.append(cl)
            ys.append(m)
            errs.append(s)
    if xs:
        ax.errorbar(xs, ys, yerr=errs,
                    label=FORMAT_LABELS[fmt],
                    color=FORMAT_COLORS[fmt],
                    marker=FORMAT_MARKERS[fmt],
                    linewidth=2, markersize=6,
                    capsize=4, capthick=1.2)

ax.set_xlabel("Number of classes")
ax.set_ylabel("Total training time (s)")
ax.set_title("Streaming — Total Training Time vs. Dataset Size\n(Tiny ImageNet, 10 epochs, batch 32)")
ax.set_xticks(TI_CLASSES)
ax.legend(framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.5)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
plt.savefig("figures/fig_streaming_ti_total_time.png", bbox_inches="tight")
plt.close()
print("Saved: figures/fig_streaming_ti_total_time.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Streaming: Total training time — Fashion MNIST (bar chart)
# ══════════════════════════════════════════════════════════════════════════════

fm_stats = get_streaming_stats("fashionmnist", 10)

fig, ax = plt.subplots(figsize=(6, 4))

fmts = [f for f in FORMATS if f in fm_stats]
means  = [fm_stats[f][0] for f in fmts]
stds   = [fm_stats[f][1] for f in fmts]
colors = [FORMAT_COLORS[f] for f in fmts]
labels = [FORMAT_LABELS[f] for f in fmts]

bars = ax.bar(labels, means, yerr=stds, color=colors,
              capsize=5, width=0.55, error_kw={"linewidth": 1.5})

for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
            f"{val:.0f}s", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Total training time (s)")
ax.set_title("Streaming — Total Training Time\n(Fashion MNIST, 60,000 images, 10 epochs, batch 32)")
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
ax.set_ylim(0, max(means) * 1.18)

plt.tight_layout()
plt.savefig("figures/fig_streaming_fm_total_time.png", bbox_inches="tight")
plt.close()
print("Saved: figures/fig_streaming_fm_total_time.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Streaming: Throughput vs number of classes (Tiny ImageNet)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 4.5))

for fmt in FORMATS:
    xs, ys, errs = [], [], []
    for cl in TI_CLASSES:
        stats = get_streaming_stats("tinyimagenet", cl, metric="training_throughput_imgs_s")
        if fmt in stats:
            m, s = stats[fmt]
            xs.append(cl)
            ys.append(m)
            errs.append(s)
    if xs:
        ax.errorbar(xs, ys, yerr=errs,
                    label=FORMAT_LABELS[fmt],
                    color=FORMAT_COLORS[fmt],
                    marker=FORMAT_MARKERS[fmt],
                    linewidth=2, markersize=6,
                    capsize=4, capthick=1.2)

ax.set_xlabel("Number of classes")
ax.set_ylabel("Throughput (images/s)")
ax.set_title("Streaming — Training Throughput vs. Dataset Size\n(Tiny ImageNet, 10 epochs, batch 32)")
ax.set_xticks(TI_CLASSES)
ax.legend(framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("figures/fig_streaming_ti_throughput.png", bbox_inches="tight")
plt.close()
print("Saved: figures/fig_streaming_ti_throughput.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Streaming: RAM usage vs number of classes (Tiny ImageNet)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 4.5))

for fmt in FORMATS:
    xs, ys, errs = [], [], []
    for cl in TI_CLASSES:
        stats = get_streaming_stats("tinyimagenet", cl, metric="ram_mb")
        if fmt in stats:
            m, s = stats[fmt]
            xs.append(cl)
            ys.append(m)
            errs.append(s)
    if xs:
        ax.errorbar(xs, ys, yerr=errs,
                    label=FORMAT_LABELS[fmt],
                    color=FORMAT_COLORS[fmt],
                    marker=FORMAT_MARKERS[fmt],
                    linewidth=2, markersize=6,
                    capsize=4, capthick=1.2)

ax.set_xlabel("Number of classes")
ax.set_ylabel("Peak RAM usage (MB)")
ax.set_title("Streaming — Peak RAM Usage vs. Dataset Size\n(Tiny ImageNet, 10 epochs, batch 32)")
ax.set_xticks(TI_CLASSES)
ax.legend(framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("figures/fig_streaming_ti_ram.png", bbox_inches="tight")
plt.close()
print("Saved: figures/fig_streaming_ti_ram.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Exp2: Load time vs number of classes (Tiny ImageNet)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 4.5))

for fmt in FORMATS:
    xs, ys, errs = [], [], []
    for cl in TI_CLASSES:
        stats = get_exp2_stats("tinyimagenet", cl, metric="load_s")
        if fmt in stats:
            m, s = stats[fmt]
            xs.append(cl)
            ys.append(m)
            errs.append(s)
    if xs:
        ax.errorbar(xs, ys, yerr=errs,
                    label=FORMAT_LABELS[fmt],
                    color=FORMAT_COLORS[fmt],
                    marker=FORMAT_MARKERS[fmt],
                    linewidth=2, markersize=6,
                    capsize=4, capthick=1.2)

ax.set_xlabel("Number of classes")
ax.set_ylabel("Load time (s)")
ax.set_title("Load Time vs. Dataset Size\n(Tiny ImageNet)")
ax.set_xticks(TI_CLASSES)
ax.legend(framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("figures/fig_exp2_ti_loadtime.png", bbox_inches="tight")
plt.close()
print("Saved: figures/fig_exp2_ti_loadtime.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Full-load vs Streaming total time comparison (50 classes TI)
# ══════════════════════════════════════════════════════════════════════════════

fl_stats = get_fullload_stats("tinyimagenet", 50)
st_stats = get_streaming_stats("tinyimagenet", 50)

fmts_fl = [f for f in FORMATS if f in fl_stats]
fmts_st = [f for f in FORMATS if f in st_stats]

x = np.arange(len(FORMATS))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))

fl_means = [fl_stats.get(f, (0, 0))[0] for f in FORMATS]
fl_stds  = [fl_stats.get(f, (0, 0))[1] for f in FORMATS]
st_means = [st_stats.get(f, (0, 0))[0] for f in FORMATS]
st_stds  = [st_stats.get(f, (0, 0))[1] for f in FORMATS]

bars1 = ax.bar(x - width/2, fl_means, width, yerr=fl_stds,
               label="Full-load", color=[FORMAT_COLORS[f] for f in FORMATS],
               capsize=4, alpha=0.9)
bars2 = ax.bar(x + width/2, st_means, width, yerr=st_stds,
               label="Streaming", color=[FORMAT_COLORS[f] for f in FORMATS],
               capsize=4, alpha=0.5, hatch="//")

ax.set_ylabel("Total training time (s)")
ax.set_title("Full-load vs. Streaming — Total Training Time\n(Tiny ImageNet, 50 classes, 10 epochs)")
ax.set_xticks(x)
ax.set_xticklabels([FORMAT_LABELS[f] for f in FORMATS])
ax.legend(framealpha=0.9)
ax.grid(True, axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("figures/fig_fullload_vs_streaming_ti50.png", bbox_inches="tight")
plt.close()
print("Saved: figures/fig_fullload_vs_streaming_ti50.png")

print("\nAll figures saved to ./figures/")
