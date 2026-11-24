# -*- coding: utf-8 -*-
"""Redraw Figure 7 panel (e): Baseline classification sanity check.

Source (manuscript Technical Validation / Baseline Classification Checks):
  Ensemble = mean of CNN + Random Forest class probabilities on the
  chronological whole-day four-class held-out test set (n = 2,020).

Artifacts (not PPT):
  validation_results/four_class/test_labels.npy
  validation_results/four_class/proba_Ensemble.npy

Plotting logic follows code-dataset/06_validation/plot_validation.py
(class-wise Precision / Recall / F1 + macro row), with Figure-6 matched
canvas and Times New Roman fonts.
"""
from __future__ import print_function

import json
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "validation_results" / "four_class"
OUT_DIR = REPO / "outputs" / HERE.name


FIGSIZE = (8.5, 6.2)  # match fig6 / fig7-d
LABEL_FS = 14
TICK_FS = 12
CELL_FS = 13
HEADER_FS = 13
TNR = Path(r"C:\Windows\Fonts\times.ttf")

COLORS4 = ["#FFC000", "#1957AD", "#70389F", "#818181"]
LABELS4 = [
    "Natural slope events",
    "Blasting",
    "Mechanical mining",
    "Transport vehicles",
]


def fp(size, weight="normal"):
    if TNR.is_file():
        return FontProperties(fname=str(TNR) if TNR.is_file() else None, family=["Times New Roman","DejaVu Serif"], size=size, weight=weight)
    return FontProperties(family="Times New Roman", size=size, weight=weight)


def load_npy(path):
    with open(str(path), "rb") as f:
        return np.load(f, allow_pickle=False)


def main():
    y = load_npy(RESULTS / "test_labels.npy")
    proba = load_npy(RESULTS / "proba_Ensemble.npy")
    pred = proba.argmax(axis=1)

    if len(y) != len(pred):
        raise ValueError("test_labels and proba_Ensemble length mismatch")
    if proba.ndim != 2 or proba.shape[1] != 4:
        raise ValueError("proba_Ensemble must be (n, 4)")

    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=range(4))
    scores = np.column_stack([prec, rec, f1])
    scores = np.vstack([scores, scores.mean(axis=0)]) * 100.0
    acc = float(accuracy_score(y, pred))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6)
    ax.axis("off")

    edges = [0.0, 1.40, 2.2666667, 3.1333333, 4.0]
    headers = ["Class", "Precision (%)", "Recall (%)", "F1-score (%)"]
    edge = "#bdbdbd"
    cmap = plt.cm.Blues
    norm = Normalize(0, 100)

    # Header row
    for j, label in enumerate(headers):
        ax.add_patch(Rectangle(
            (edges[j], 5), edges[j + 1] - edges[j], 1,
            facecolor="white", edgecolor=edge, lw=0.7,
        ))
        ax.text(
            (edges[j] + edges[j + 1]) / 2, 5.5, label,
            ha="center", va="center", fontproperties=fp(HEADER_FS),
        )

    row_labels = LABELS4 + ["Overall (macro)"]
    for i, lab in enumerate(row_labels):
        yy = 4 - i
        bold = "bold" if i == 4 else "normal"
        ax.add_patch(Rectangle(
            (0, yy), 1.4, 1, facecolor="white", edgecolor=edge, lw=0.7,
        ))
        if i < 4:
            ax.scatter(0.12, yy + 0.5, s=90, color=COLORS4[i], zorder=3)
            ax.text(
                0.24, yy + 0.5, lab, ha="left", va="center",
                fontproperties=fp(TICK_FS, weight=bold),
            )
        else:
            ax.text(
                0.70, yy + 0.5, lab, ha="center", va="center",
                fontproperties=fp(TICK_FS, weight=bold),
            )

        for j, val in enumerate(scores[i]):
            ax.add_patch(Rectangle(
                (edges[j + 1], yy), edges[j + 2] - edges[j + 1], 1,
                facecolor=cmap(norm(val)), edgecolor=edge, lw=0.7,
            ))
            ax.text(
                (edges[j + 1] + edges[j + 2]) / 2, yy + 0.5,
                "{:.3f}".format(val),
                ha="center", va="center",
                color="white" if val > 65 else "black",
                fontproperties=fp(CELL_FS, weight=bold),
            )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03, shrink=0.72)
    cbar.ax.set_title("Score\n(%)", fontproperties=fp(11), pad=6)
    cbar.set_ticks([0, 50, 100])
    for t in cbar.ax.get_yticklabels():
        t.set_fontproperties(fp(TICK_FS))
    cbar.outline.set_edgecolor(edge)

    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.08)
    fig.suptitle(
        "Baseline classification sanity check  |  Ensemble (CNN + RF)  |  "
        "n = {:,} test records  |  Acc = {:.3f}".format(len(y), acc),
        fontproperties=fp(LABEL_FS), y=0.97,
    )

    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / "fig7-e.pdf"
    png = out / "fig7-e.png"
    fig.savefig(str(pdf), dpi=300)
    fig.savefig(str(png), dpi=300)
    plt.close(fig)

    report = {
        "n_test": int(len(y)),
        "accuracy": acc,
        "per_class_scores_percent": {
            name: [float(x) for x in row]
            for name, row in zip(row_labels, scores)
        },
        "source": {
            "test_labels": str(RESULTS / "test_labels.npy"),
            "proba_Ensemble": str(RESULTS / "proba_Ensemble.npy"),
        },
    }
    with open(str(out / "fig7e_values.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


    print("wrote", pdf)
    print("wrote", png)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    OUT_DIR = args.out
    RESULTS = args.results
    main()
