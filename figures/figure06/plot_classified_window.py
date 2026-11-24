# -*- coding: utf-8 -*-
"""Plot a classified blasting .npy record for Figure 6(b).

Per manuscript Data Records / Fig. 6(b):
  - signal: vertical-component amplitude after 1--20 Hz band-pass, instrument counts
  - time_axis: seconds from the first sample of the nominal 10 s window
  - sampling_rate: 250 Hz
  - window: 2 s pre-trigger + 8 s post-trigger

Canvas size matches fig6-a (8.5 x 6.2 in)
for a later one-row, two-column manuscript layout.
"""
from __future__ import print_function
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent / "input"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figure06"
NPY = ROOT / "sensor_230000081_20250603T103254.752000_20250603T103257.320000.npy"
OUT_PDF = OUT_DIR / "fig6-b.pdf"
OUT_PNG = OUT_DIR / "fig6-b.png"

# Match the three-component raw-waveform figure exactly
FIGSIZE = (8.5, 6.2)
BLUE = "#1F77B4"
TNR = Path(r"C:\Windows\Fonts\times.ttf")

# Paper-readable sizes when later placed at ~half text width
TITLE_FS = 13
LABEL_FS = 14
TICK_FS = 12
NOTE_FS = 11


def fp(size):
    if TNR.is_file():
        return FontProperties(fname=str(TNR) if TNR.is_file() else None, family=["Times New Roman","DejaVu Serif"], size=size, weight="normal")
    return FontProperties(family="Times New Roman", size=size, weight="normal")


def main():
    rec = np.load(str(NPY), allow_pickle=True).item()
    t = np.asarray(rec["time_axis"], dtype=float)          # seconds
    y = np.asarray(rec["signal"], dtype=float)             # instrument counts
    fs = float(rec["sampling_rate"])
    sid = str(rec["sensor_id"])
    t0 = str(rec["event_start"])
    t1 = str(rec["event_end"])

    if len(t) != len(y):
        raise RuntimeError("time_axis and signal length mismatch")
    # Sanity from data (not assumed): duration and sample rate
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0 / fs
    dur = float(t[-1] - t[0] + dt)

    # Trigger onset is 2 s after window start per manuscript; verify vs metadata span.
    # event_end - event_start from ISO strings for annotation only.
    from datetime import datetime
    fmt = "%Y-%m-%dT%H:%M:%S.%f"
    # stored without timezone separators inconsistently; support both
    def parse(s):
        s = s.replace("Z", "")
        if "T" not in s:
            raise ValueError(s)
        # allow missing microseconds padding
        if "." in s:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

    # filename uses T without dashes in date part already handled in fields
    try:
        ts0 = parse(t0)
        ts1 = parse(t1)
        trig_len = (ts1 - ts0).total_seconds()
    except Exception:
        trig_len = None

    # Per paper: first sample precedes event_start by 2 s
    trig_on = 2.0
    trig_off = trig_on + trig_len if trig_len is not None else None

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t, y, color=BLUE, lw=1.0, solid_capstyle="butt")

    # Optional trigger interval shade (data-derived), does not alter axes units
    if trig_off is not None:
        ax.axvspan(trig_on, trig_off, color="#1F77B4", alpha=0.08, lw=0, zorder=0)
        ax.axvline(trig_on, color="#555555", lw=0.8, ls="--", zorder=2)
        ax.axvline(trig_off, color="#555555", lw=0.8, ls="--", zorder=2)

    ax.set_xlim(float(t[0]), float(t[-1]))
    peak = float(np.max(np.abs(y)))
    ax.set_ylim(-1.15 * peak, 1.15 * peak)

    ax.set_xlabel("Time (s)", fontproperties=fp(LABEL_FS))
    ax.set_ylabel("Amplitude (counts)", fontproperties=fp(LABEL_FS))
    ax.tick_params(width=0.9, length=4.0, labelsize=TICK_FS)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(fp(TICK_FS))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    title = ("Classified blasting waveform (vertical, 1--20 Hz)  |  "
             "Station {}  |  {} Hz  |  {:.3f} s window").format(sid, int(round(fs)), dur)
    fig.suptitle(title, fontproperties=fp(TITLE_FS), y=0.96)

    note = ("event_start = {}   event_end = {}   "
            "n = {} samples").format(t0, t1, len(y))
    fig.text(0.12, 0.02, note, ha="left", va="bottom",
             fontproperties=fp(NOTE_FS), color="#333333")

    # Fixed canvas (no tight crop) so page size matches the raw 3-component figure
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.14)
    fig.savefig(str(OUT_PDF), dpi=300)
    fig.savefig(str(OUT_PNG), dpi=300)
    plt.close(fig)

    print("wrote", OUT_PDF)
    print("wrote", OUT_PNG)
    print("sensor_id", sid)
    print("sampling_rate_Hz", fs)
    print("n_samples", len(y))
    print("time_axis_s: [{:.6f}, {:.6f}]".format(t[0], t[-1]))
    print("signal_counts: min={:.6f} max={:.6f}".format(y.min(), y.max()))
    print("figsize_in", FIGSIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    OUT_PDF = args.out / OUT_PDF.name
    OUT_PNG = args.out / OUT_PNG.name
    main()
