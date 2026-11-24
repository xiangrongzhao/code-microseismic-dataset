"""Rebuild portable NPZ records and their index from the trusted original dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from load_event import FIELDS, INDEX_FIELDS, TEXT_FIELDS, validate_event


def build(dataset_root, output_root, trust_source=False):
    if not trust_source:
        raise ValueError("Conversion reads the original pickled NPY dictionaries; explicitly trust the source.")
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    source = dataset_root / "classified event-level waveform data"
    destination = output_root / "classified event-level waveform data (npz)"
    index_path = output_root / "records_index.csv"
    report_path = output_root / "conversion_report.json"
    files = sorted(source.rglob("*.npy"))
    if not files:
        raise ValueError(f"No source NPY files found: {source}")
    if destination.exists() or index_path.exists() or report_path.exists():
        raise FileExistsError("Use a fresh output directory; existing release data will not be overwritten.")
    output_root.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    rows = []
    source_bytes = 0
    npz_bytes = 0
    for number, path in enumerate(files, 1):
        relative = path.relative_to(source)
        if len(relative.parts) != 2:
            raise ValueError(f"Expected one class folder per record: {relative}")
        original_bytes = path.read_bytes()
        source_bytes += len(original_bytes)
        original = np.load(io.BytesIO(original_bytes), allow_pickle=True).item()
        validate_event(original)
        portable = {name: np.asarray(original[name], dtype=np.str_) for name in TEXT_FIELDS}
        for name in ("time_axis", "signal", "sampling_rate"):
            portable[name] = np.asarray(original[name], dtype="<f8")
        validate_event(portable, portable=True)
        target = destination / relative.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            np.savez_compressed(handle, **{name: portable[name] for name in FIELDS})
        saved_bytes = target.read_bytes()
        npz_bytes += len(saved_bytes)
        with np.load(io.BytesIO(saved_bytes), allow_pickle=False) as saved:
            if set(saved.files) != set(FIELDS) or len(saved.files) != len(FIELDS):
                raise ValueError(f"Unexpected saved fields: {target}")
            roundtrip = {name: saved[name] for name in FIELDS}
        validate_event(roundtrip, portable=True)
        for name in FIELDS:
            if not np.array_equal(np.asarray(original[name]), roundtrip[name]):
                raise ValueError(f"Round-trip mismatch in {name}: {path}")
        rows.append({
            "record_id": relative.with_suffix("").as_posix(),
            "class": relative.parts[0],
            "npy_path": path.relative_to(dataset_root).as_posix(),
            "npz_path": target.relative_to(output_root).as_posix(),
            **{name: str(np.asarray(original[name]).item()) for name in TEXT_FIELDS},
            "sampling_rate": repr(float(original["sampling_rate"])),
            "n_samples": len(original["signal"]),
            "npy_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "npz_sha256": hashlib.sha256(saved_bytes).hexdigest(),
        })
        if number % 2000 == 0:
            print(f"Converted and compared {number}/{len(files)} records", flush=True)
    temporary_index = index_path.with_suffix(".csv.tmp")
    with temporary_index.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_index, index_path)
    report = {
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": source.name, "destination_directory": destination.name,
        "source_npy_records": len(files), "npz_records": len(rows), "index_rows": len(rows),
        "npz_reopened_with_allow_pickle_false": len(rows),
        "records_compared_exactly_in_all_six_fields": len(rows),
        "field_comparisons_passed": len(rows) * len(FIELDS),
        "source_npy_bytes": source_bytes, "npz_bytes": npz_bytes,
        "classes": dict(sorted(Counter(row["class"] for row in rows).items())),
        "numpy_version": np.__version__,
        "records_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "conversion": "Lossless field serialization only; no filtering, resampling, amplitude scaling or relabelling.",
        "npy_path_root": "Original dataset root supplied to --dataset-root",
        "npz_path_root": "Directory containing records_index.csv",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Must not already contain release data")
    parser.add_argument("--trust-source", action="store_true")
    args = parser.parse_args()
    build(args.dataset_root, args.output_root, args.trust_source)
