"""Plot the station-by-date matrix underlying Figure 3."""
import argparse
import csv
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def draw(matrix, destination):
    with matrix.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        dates = [date.fromisoformat(s) for s in next(reader)[1:]]
        rows = list(reader)
    values = np.asarray([r[1:] for r in rows], dtype=int)
    if values.shape != (36, 50) or not np.isin(values, (0, 1)).all():
        raise ValueError("Expected a binary matrix for 36 stations and 50 dates.")
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    ax.imshow(values, cmap=ListedColormap(["#dbdbdb", "#2ca02c"]), vmin=0, vmax=1,
              origin="upper", aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(rows)), labels=[r[0] for r in rows], fontsize=8)
    ticks = [0, 14, 30, 44, 49]
    ax.set_xticks(ticks, labels=[dates[i].strftime("%b %d") for i in ticks])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(.5, len(rows), 1), minor=True)
    ax.grid(which="minor", axis="y", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    ax.legend(handles=[Patch(color="#2ca02c", label="Data exist"),
                       Patch(color="#dbdbdb", label="Data missing")],
              loc="lower center", bbox_to_anchor=(.5, 1.03), ncol=2, frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--matrix", type=Path, default=Path(__file__).with_name("availability_by_station_and_date.csv"))
    args = parser.parse_args()
    draw(args.matrix, args.output)
