# -*- coding: utf-8 -*-
"""Visualize three-component 5-min miniSEED raw waveforms (Figure 6(a)).

Paper: panel (a) shows five-minute segmentation of three-component raw records
(DHN, DHE, DHZ), sample rate 250 Hz, Time (min) on the horizontal axis.
Filenames swap DHE/DHN relative to miniSEED headers; we label by header channel.
"""
from __future__ import print_function
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from obspy import read

ROOT = Path(__file__).resolve().parent / "input"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figure06"
OUT_PDF = OUT_DIR / "fig6-a.pdf"
OUT_PNG = OUT_DIR / "fig6-a.png"
TNR = Path(r"C:\Windows\Fonts\times.ttf")
BLUE = "#1F77B4"


def fp(size):
    if TNR.is_file():
        return FontProperties(fname=str(TNR) if TNR.is_file() else None, family=["Times New Roman","DejaVu Serif"], size=size, weight="normal")
    return FontProperties(family="Times New Roman", size=size, weight="normal")


def load_components(folder):
    """Return ordered traces DHN, DHE, DHZ keyed by miniSEED header channel."""
    by_chan = {}
    for path in sorted(folder.glob("*.miniseed")):
        tr = read(str(path))[0]
        by_chan[tr.stats.channel.upper()] = (path.name, tr)
    order = ["DHN", "DHE", "DHZ"]
    missing = [c for c in order if c not in by_chan]
    if missing:
        raise RuntimeError("missing channels: {} in {}".format(missing, folder))
    return [(c, by_chan[c][0], by_chan[c][1]) for c in order]


def main():
    components = load_components(ROOT)
    # Shared time axis in minutes from segment start (5-min miniSEED).
    _, _, tr0 = components[0]
    npts = tr0.stats.npts
    dt = tr0.stats.delta
    t_min = np.arange(npts, dtype=float) * dt / 60.0
    sr = tr0.stats.sampling_rate
    station = "230000081"
    start = str(tr0.stats.starttime)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Same canvas as classified blasting panel for 1-row 2-column layout
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.2), sharex=True)
    for ax, (chan, fname, tr) in zip(axes, components):
        data = tr.data.astype(float)
        ax.plot(t_min, data, color=BLUE, lw=0.55, solid_capstyle="butt")
        ax.set_ylabel("Amplitude (counts)", fontproperties=fp(14))
        ax.set_xlim(0.0, 5.0)
        peak = np.max(np.abs(data))
        ax.set_ylim(-1.15 * peak, 1.15 * peak)
        ax.tick_params(width=0.9, length=4.0, labelsize=12)
        for lab in ax.get_yticklabels():
            lab.set_fontproperties(fp(12))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.text(0.012, 0.90, chan, transform=ax.transAxes, ha="left", va="top",
                fontproperties=fp(13), color="#1a1a1a")

    axes[-1].set_xlabel("Time (min)", fontproperties=fp(14))
    for lab in axes[-1].get_xticklabels():
        lab.set_fontproperties(fp(12))

    title = ("Raw three-component miniSEED waveform  |  Station {}  |  "
             "Start {}  |  {} Hz  |  5 min").format(station, start, int(sr))
    fig.suptitle(title, fontproperties=fp(13), y=0.96)
    # Fixed canvas (no tight crop) — identical page size to blasting figure
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.10, hspace=0.14)

    fig.savefig(str(OUT_PDF), dpi=300)
    fig.savefig(str(OUT_PNG), dpi=300)
    plt.close(fig)
    print("wrote", OUT_PDF)
    print("wrote", OUT_PNG)
    for chan, fname, tr in components:
        print(chan, "file=", fname, "header=", tr.stats.channel,
              "npts=", tr.stats.npts, "duration_s=", tr.stats.npts * tr.stats.delta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    OUT_PDF = args.out / OUT_PDF.name
    OUT_PNG = args.out / OUT_PNG.name
    main()
