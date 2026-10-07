"""Paths and array checks shared by the localization reproduction scripts."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def prepare_output(path):
    """Keep generated inputs and results outside the released source and dataset."""
    out = Path(path).resolve()
    protected = [HERE]
    if HERE.parent.name == "code-dataset":
        protected.extend([HERE.parent, HERE.parent.parent / "dataset"])
    for root in protected:
        root = root.resolve()
        if out == root or out.is_relative_to(root):
            raise ValueError("Choose an output directory outside code-dataset and dataset.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_example_inputs(directory):
    base = Path(directory).resolve()
    report = pd.read_csv(base / "pick_report.csv", dtype={"sid": str})
    stations = np.load(base / "stas_xyz.npy", allow_pickle=False).astype(float)
    arrivals = np.load(base / "tobs_p.npy", allow_pickle=False).astype(float)
    if arrivals.ndim != 1 or stations.shape != (len(arrivals), 3) or len(arrivals) < 5:
        raise ValueError("Expected at least five arrivals and matching (N, 3) station coordinates.")
    if not np.isfinite(stations).all() or not np.isfinite(arrivals).all():
        raise ValueError("Station coordinates and arrivals must be finite.")
    if len(report) != len(arrivals) or report.sid.duplicated().any():
        raise ValueError("The pick report must contain one row per distinct station.")
    if not np.allclose(report[["x_km", "y_km", "z_km"]].to_numpy(float), stations):
        raise ValueError("The pick report and station array do not have the same row order.")
    if not np.allclose(report.tobs_p.to_numpy(float), arrivals):
        raise ValueError("The pick report and arrival array do not have the same row order.")
    return report, stations, arrivals


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
