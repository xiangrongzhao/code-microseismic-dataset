"""Build the localization example inputs from the classified waveform records.

The example event is the record group labelled as a small landslide on 2025-06-09 at 22:01:52 UTC,
recorded by seven southern-highwall stations. P onsets are picked automatically so that the inputs
can be regenerated from the waveforms without any manual step.

By default the records are read from inputs/waveforms/ in this directory, which holds the seven records of
the event copied unchanged from the small-landslide category of the dataset. A different directory,
for example the small-landslide directory of a downloaded copy of the dataset, can be given as the
--records argument; files are matched by station and trigger time.

Outputs, written to the external --out-dir directory:
    stas_xyz.npy      (n, 3) station coordinates (x, y, z) in km, same frame as the velocity model
    tobs_p.npy        (n,)   P arrival times in seconds relative to the earliest pick
    pick_report.csv          one row per station: file, absolute pick time, signal-to-noise ratio,
                             coordinates

Usage:
    python make_inputs1.py --out-dir <external input directory> [--records <record directory>]
"""
import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from reproduction_io import prepare_output, sha256

HERE = Path(__file__).resolve().parent
STATIONS = HERE / "inputs" / "stations.csv"

EVENT_WINDOW = (datetime(2025, 6, 9, 22, 1, 48), datetime(2025, 6, 9, 22, 2, 0))

FS = 250.0            # sampling rate of the released records, Hz
PRE_TRIGGER_S = 2.0   # each record starts 2 s before the trigger onset
BAND_HZ = (2.0, 20.0)
PICK_WINDOW_S = (1.0, 3.5)
MIN_SNR = 3.0

SOS = butter(4, BAND_HZ, btype="band", fs=FS, output="sos")
# sensor_<id>_<YYYYMMDDTHHMMSS.ffffff>_<YYYYMMDDTHHMMSS.ffffff>.npy
NAME = re.compile(r"sensor_(\d{9})_(\d{8}T\d{6})[._](\d{6})_")


def aic_pick(x):
    """Akaike information criterion pick: the sample that best splits noise from signal."""
    n = len(x)
    best, kb = np.inf, None
    for k in range(10, n - 10):
        v1, v2 = np.var(x[:k]), np.var(x[k:])
        if v1 > 0 and v2 > 0:
            a = k * np.log(v1) + (n - k - 1) * np.log(v2)
            if a < best:
                best, kb = a, k
    return kb


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=HERE / "inputs" / "waveforms")
    parser.add_argument("--stations", type=Path, default=STATIONS)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.records.is_dir():
        parser.error(f"record directory not found: {args.records}")
    out = prepare_output(args.out_dir)
    if out == args.records.resolve() or out.is_relative_to(args.records.resolve()):
        parser.error("Generated inputs must be outside the waveform directory.")
    sta = pd.read_csv(args.stations, dtype={"station_id": str}).set_index("station_id")
    rows = []
    record_hashes = {}
    for f in sorted(args.records.glob("sensor_*.npy")):
        m = NAME.match(f.name)
        if not m:
            continue
        trig = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S") + timedelta(microseconds=int(m.group(3)))
        if not (EVENT_WINDOW[0] <= trig <= EVENT_WINDOW[1]) or m.group(1) not in sta.index:
            continue
        rec = np.load(f, allow_pickle=True).item()
        sig = np.asarray(rec["signal"], float)
        if (str(rec["sensor_id"]) != m.group(1)
                or not np.isclose(float(rec["sampling_rate"]), FS, rtol=1e-6, atol=1e-6)
                or sig.ndim != 1 or len(sig) not in (2499, 2500)
                or np.asarray(rec["time_axis"]).shape != sig.shape or not np.isfinite(sig).all()):
            raise ValueError(f"Invalid station, sample rate or waveform shape: {f.name}")
        y = sosfiltfilt(SOS, sig - sig.mean())
        i0 = int(PICK_WINDOW_S[0] * FS)
        picked = aic_pick(y[i0:int(PICK_WINDOW_S[1] * FS)])
        if picked is None:
            raise ValueError(f"No AIC onset found: {f.name}")
        k = picked + i0
        noise = np.std(y[max(k - int(FS), 0):k]) + 1e-12
        snr = float(np.abs(y[k:k + int(0.4 * FS)]).max() / noise)
        t_pick = (datetime.fromisoformat(str(rec["event_start"])) - timedelta(seconds=PRE_TRIGGER_S)
                  + timedelta(seconds=k / FS))
        s = sta.loc[m.group(1)]
        rows.append(dict(sid=m.group(1), file=f.name, t_abs_p=t_pick, snr=round(snr, 2),
                         x_km=s.x_km, y_km=s.y_km, z_km=s.z_km))
        record_hashes[f.name] = sha256(f)

    df = pd.DataFrame(rows)
    print(f"{len(df)} records in the event window")
    if not rows:
        raise ValueError("No matching waveform records were found in the event window.")
    df = df[df.snr >= MIN_SNR].sort_values("t_abs_p").reset_index(drop=True)
    if len(df) < 5 or df.sid.duplicated().any():
        raise ValueError("At least five distinct stations must pass the SNR threshold.")
    df["tobs_p"] = (df.t_abs_p - df.t_abs_p.min()).dt.total_seconds().round(4)

    np.save(out / "stas_xyz.npy", df[["x_km", "y_km", "z_km"]].to_numpy(float))
    np.save(out / "tobs_p.npy", df.tobs_p.to_numpy(float))
    df[["sid", "file", "t_abs_p", "snr", "x_km", "y_km", "z_km", "tobs_p"]].to_csv(
        out / "pick_report.csv", index=False)
    (out / "input_provenance.json").write_text(json.dumps({
        "station_table_sha256": sha256(args.stations),
        "waveform_sha256": {name: record_hashes[name] for name in df.file},
        "sampling_rate_hz": FS, "band_hz": BAND_HZ, "pick_window_s": PICK_WINDOW_S,
        "pre_trigger_s": PRE_TRIGGER_S, "minimum_snr": MIN_SNR,
        "output_sha256": {name: sha256(out / name)
                          for name in ("stas_xyz.npy", "tobs_p.npy", "pick_report.csv")},
    }, indent=2), encoding="utf-8")
    print(f"{len(df)} picks with a signal-to-noise ratio of at least {MIN_SNR:.0f} written to {out}")
    print(df[["sid", "t_abs_p", "snr", "tobs_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
