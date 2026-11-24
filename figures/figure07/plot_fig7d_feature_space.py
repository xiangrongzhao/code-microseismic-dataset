# -*- coding: utf-8 -*-
"""Redraw Figure 7 panel (d): feature-space t-SNE inspection.

Data logic follows code-dataset/06_validation/plot_validation.py::draw_feature_space
(uses saved tsne_projection.csv; does not refit t-SNE).

Canvas and fonts match Figure 6 subplots:
  figsize = (8.5, 6.2) in, Times New Roman, label 14 / tick 12 / legend 12.
"""
from __future__ import print_function

import json
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "validation_results" / "four_class"
OUT_DIR = REPO / "outputs" / HERE.name


FIGSIZE = (8.5, 6.2)  # match fig6
LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 12
TNR = Path(r"C:\Windows\Fonts\times.ttf")

COLORS4 = ["#FFC000", "#1957AD", "#70389F", "#818181"]
LABELS4 = [
    "Natural slope events",
    "Blasting",
    "Mechanical mining",
    "Transport vehicles",
]


def fp(size):
    if TNR.is_file():
        return FontProperties(fname=str(TNR) if TNR.is_file() else None, family=["Times New Roman","DejaVu Serif"], size=size, weight="normal")
    return FontProperties(family="Times New Roman", size=size, weight="normal")


def main():
    results = RESULTS
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    def read_text(path):
        # Old pandas on Windows fails on non-ASCII paths; read via handles.
        with open(str(path), "r", encoding="utf-8") as f:
            return f.read()

    def read_csv(path):
        with open(str(path), "rb") as f:
            return pd.read_csv(f)

    info = json.loads(read_text(results / "summary.json"))
    if info["train_schema"] != "4class":
        raise ValueError("Panel d requires a completed four-class run.")

    proj = read_csv(results / "tsne_projection.csv")
    split = read_csv(results / "splits.csv")
    with open(str(results / "valid_mask.npy"), "rb") as f:
        valid = np.load(f, allow_pickle=False)
    record_indices = proj.record_index.values.astype(int)
    labels = np.array([0, 1, 0, 2, 3])[split.class_idx.values.astype(int)]

    if (not valid.all() or len(valid) != len(split)
            or not proj.record_index.is_unique
            or (record_indices < 0).any() or (record_indices >= len(split)).any()
            or not np.isfinite(proj[["tsne_x", "tsne_y"]].values).all()
            or not np.array_equal(labels[record_indices], proj.class_index.values.astype(int))):
        raise ValueError("The projection does not match the classification records.")
    stats = json.loads(read_text(results / "feature_statistics.json"))
    if len(proj) != stats["tsne_subsample"]:
        raise ValueError("The projection sample count does not match its metadata.")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for c, (label, color) in enumerate(zip(LABELS4, COLORS4)):
        v = proj[proj.class_index == c]
        ax.scatter(
            v.tsne_x, v.tsne_y, s=14, alpha=0.65, color=color,
            label=label, edgecolors="none",
        )

    ax.set_xlabel("t-SNE Dimension 1", fontproperties=fp(LABEL_FS))
    ax.set_ylabel("t-SNE Dimension 2", fontproperties=fp(LABEL_FS))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0.035, y=0.04)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    # Legend inside axes, lower-left, 2 x 2
    leg = ax.legend(
        loc="lower left",
        ncol=2,
        frameon=False,
        markerscale=2.0,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.55,
        borderpad=0.5,
        prop=fp(LEGEND_FS),
    )
    for text in leg.get_texts():
        text.set_fontproperties(fp(LEGEND_FS))

    # Fixed canvas (no tight crop) — same PDF page as fig6: 8.5 x 6.2 in
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.12, top=0.94)

    pdf = out / "fig7-d.pdf"
    png = out / "fig7-d.png"
    fig.savefig(str(pdf), dpi=300)
    fig.savefig(str(png), dpi=300)
    plt.close(fig)


    print("wrote", pdf)
    print("wrote", png)
    print("n_points", len(proj))
    print("figsize_in", FIGSIZE)
    print("fonts", "label", LABEL_FS, "tick", TICK_FS, "legend", LEGEND_FS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    OUT_DIR = args.out
    RESULTS = args.results
    main()
