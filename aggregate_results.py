"""
aggregate_results.py
────────────────────────────────────────────────────────────────────────────
Reads a JSON results file produced by benchmark.py and prints a summary
table with mean ± std for accuracy and median for timing metrics.

Usage:
  python aggregate_results.py results_streaming_tinyimagenet_50classes.json
  python aggregate_results.py results_fullload_mnist.json
"""

import json
import sys

import numpy as np


def _print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    sep    = "─" * (sum(widths) + 2 * len(widths) + 2)
    line   = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(sep)
    print(line.format(*headers))
    print(sep)
    for r in rows:
        print(line.format(*r))
    print(sep)


def aggregate(filepath: str):
    with open(filepath) as f:
        runs = json.load(f)

    if not runs:
        print("No results found.")
        return

    mode    = runs[0]["mode"]
    dataset = runs[0]["dataset"]
    mc      = runs[0].get("max_classes")

    print(f"\nFile   : {filepath}")
    print(f"Mode   : {mode}  |  Dataset: {dataset}" + (f"  |  Classes: {mc}" if mc else ""))
    print(f"Entries: {len(runs)}\n")

    # Group metrics by format
    by_fmt: dict = {}
    for run in runs:
        for fmt, metrics in run["results"].items():
            by_fmt.setdefault(fmt, []).append(metrics)

    def med(entries, key):
        vals = [e[key] for e in entries if key in e]
        return f"{np.median(vals):.3f}" if vals else "—"

    def mean_f(entries, key):
        vals = [e[key] for e in entries if key in e]
        return f"{np.mean(vals):.1f}" if vals else "—"

    def acc_str(entries):
        accs = [e["accuracy"] for e in entries]
        if len(accs) > 1:
            return f"{np.mean(accs):.4f} ± {np.std(accs):.4f}"
        return f"{accs[0]:.4f}"

    def conv_str(entries):
        vals = [e["convert_s"] for e in entries]
        if not any(vals):   # source format: all zeros
            return "—"
        return f"{np.median(vals):.3f}"

    fmt_order = ["imagefolder", "hdf5", "npz", "tfrecord"]

    if mode == "streaming":
        headers = ["Format", "Runs", "Convert(s)", "1stBatch(s)", "RAM(MB)",
                   "File(MB)", "Train(s)", "Samples/s", "Accuracy"]
        rows = []
        for fmt in fmt_order:
            if fmt not in by_fmt:
                continue
            e = by_fmt[fmt]
            rows.append([
                fmt, str(len(e)),
                conv_str(e), med(e, "first_batch_s"),
                mean_f(e, "ram_mb"), mean_f(e, "file_mb"),
                med(e, "train_s"), med(e, "samples_s"),
                acc_str(e),
            ])
    else:
        headers = ["Format", "Runs", "Convert(s)", "Load(s)", "RAM(MB)",
                   "File(MB)", "Train(s)", "Accuracy"]
        rows = []
        for fmt in fmt_order:
            if fmt not in by_fmt:
                continue
            e = by_fmt[fmt]
            rows.append([
                fmt, str(len(e)),
                conv_str(e), med(e, "load_s"),
                mean_f(e, "ram_mb"), mean_f(e, "file_mb"),
                med(e, "train_s"),
                acc_str(e),
            ])

    _print_table(rows, headers)
    print("  Timing: median across runs  |  RAM/File: mean across runs")
    if len(next(iter(by_fmt.values()))) > 1:
        print("  Accuracy: mean ± std across runs")
    else:
        print("  Accuracy: single run (use --runs 3+ for mean ± std)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python aggregate_results.py <results_file.json>")
        sys.exit(1)
    aggregate(sys.argv[1])
