"""Load and verify the indexed, pickle-free classified event release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
FIELDS = ("sensor_id", "event_start", "event_end", "time_axis", "signal", "sampling_rate")
TEXT_FIELDS = FIELDS[:3]
INDEX_FIELDS = (
    "record_id", "class", "npy_path", "npz_path", "sensor_id", "event_start",
    "event_end", "sampling_rate", "n_samples", "npy_sha256", "npz_sha256",
)


def checked_path(root, relative):
    """Resolve a portable index path inside its documented root."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError(f"Index path must be relative and remain inside its root: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Index path escapes its root: {relative}")
    return path


def read_index(index_path=None):
    path = Path(index_path or PACKAGE_ROOT / "records_index.csv").resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(INDEX_FIELDS):
            raise ValueError("Unexpected record-index columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("The record index is empty.")
    for name in ("record_id", "npz_path", "npy_path"):
        if len({row[name] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {name} in the record index.")
    for row in rows:
        for name in ("npy_sha256", "npz_sha256"):
            digest = row[name]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid {name} for {row['record_id']}")
    return path, rows


def verified_bytes(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return data


def validate_event(value, portable=False):
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise ValueError("Expected exactly the six documented event fields.")
    for name in TEXT_FIELDS:
        a = np.asarray(value[name])
        if a.ndim != 0 or a.dtype.kind != "U":
            raise ValueError(f"{name} must be a scalar Unicode string.")
    for name in ("signal", "time_axis"):
        a = np.asarray(value[name])
        if a.ndim != 1 or not len(a) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError(f"{name} must be a finite, non-empty numeric vector.")
        if portable and (a.dtype.kind != "f" or a.dtype.itemsize != 8):
            raise ValueError(f"{name} must use float64 in the NPZ release.")
    time = np.asarray(value["time_axis"])
    if len(time) != len(value["signal"]):
        raise ValueError("Waveform and time-axis lengths differ.")
    fs_array = np.asarray(value["sampling_rate"])
    if fs_array.ndim != 0 or fs_array.dtype.kind not in "fiu":
        raise ValueError("sampling_rate must be a numeric scalar.")
    fs = float(fs_array)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_rate must be finite and positive.")
    if portable and (fs_array.dtype.kind != "f" or fs_array.dtype.itemsize != 8):
        raise ValueError("sampling_rate must use float64 in the NPZ release.")
    if len(time) > 1 and not np.allclose(np.diff(time), 1 / fs, rtol=1e-6, atol=1e-10):
        raise ValueError("Time-axis spacing does not match sampling_rate.")


def check_metadata(value, row):
    for name in TEXT_FIELDS:
        if str(np.asarray(value[name]).item()) != row[name]:
            raise ValueError(f"Index metadata differs from {name}: {row['record_id']}")
    if float(value["sampling_rate"]) != float(row["sampling_rate"]):
        raise ValueError(f"Index sampling_rate differs: {row['record_id']}")
    if len(value["signal"]) != int(row["n_samples"]):
        raise ValueError(f"Index n_samples differs: {row['record_id']}")


def load_indexed(row, index_root, dataset_root=None, use_npy=False, trust_source=False):
    """A trusted index checks integrity; it does not make pickle trustworthy."""
    if use_npy:
        if not trust_source:
            raise ValueError("Legacy NPY requires explicit trust_source=True / --trust-source.")
        root = Path(dataset_root) if dataset_root else Path(index_root).parent.parent / "dataset"
        path = checked_path(root, row["npy_path"])
        data = verified_bytes(path, row["npy_sha256"])
        value = np.load(io.BytesIO(data), allow_pickle=True).item()
    else:
        path = checked_path(index_root, row["npz_path"])
        data = verified_bytes(path, row["npz_sha256"])
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if len(archive.files) != len(FIELDS) or set(archive.files) != set(FIELDS):
                raise ValueError(f"Unexpected NPZ fields: {path}")
            value = {name: archive[name] for name in FIELDS}
    validate_event(value, portable=not use_npy)
    check_metadata(value, row)
    return {
        **{name: str(np.asarray(value[name]).item()) for name in TEXT_FIELDS},
        "time_axis": np.asarray(value["time_axis"]),
        "signal": np.asarray(value["signal"]),
        "sampling_rate": float(value["sampling_rate"]),
    }


def load_event(record, *, index_path=None, dataset_root=None, use_npy=False, trust_source=False):
    """Return a six-field dictionary after checking the indexed SHA-256.

    record is a record_id (class/file_stem), an indexed relative NPZ path,
    or the absolute path of an indexed NPZ file. NPZ is always the default.
    """
    index, rows = read_index(index_path)
    record = str(record).replace("\\", "/")
    matches = [row for row in rows if record in (row["record_id"], row["npz_path"])]
    if not matches and Path(record).is_absolute():
        target = Path(record).resolve()
        matches = [row for row in rows if checked_path(index.parent, row["npz_path"]) == target]
    if len(matches) != 1:
        raise ValueError("Record is absent or ambiguous; use the exact record_id from records_index.csv.")
    return load_indexed(matches[0], index.parent, dataset_root, use_npy, trust_source)


def verify_all(index_path=None, dataset_root=None, include_npy=False):
    """Verify every NPZ; optionally hash original NPY files without unpickling."""
    index, rows = read_index(index_path)
    expected = {checked_path(index.parent, row["npz_path"]) for row in rows}
    actual = set((index.parent / "classified event-level waveform data (npz)").rglob("*.npz"))
    if actual != expected:
        raise ValueError(f"NPZ inventory differs from index: missing={len(expected-actual)}, extra={len(actual-expected)}")
    npy_root = Path(dataset_root) if dataset_root else index.parent.parent.parent / "dataset"
    for number, row in enumerate(rows, 1):
        load_indexed(row, index.parent)
        if include_npy:
            verified_bytes(checked_path(npy_root, row["npy_path"]), row["npy_sha256"])
        if number % 2000 == 0:
            print(f"Verified {number}/{len(rows)} records", flush=True)
    return {
        "status": "passed", "records": len(rows),
        "npz_sha256_schema_and_metadata_verified": len(rows),
        "npy_sha256_verified": len(rows) if include_npy else 0,
        "classes": dict(sorted(Counter(row["class"] for row in rows).items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", nargs="?", help="record_id or indexed NPZ path")
    parser.add_argument("--index", type=Path, default=PACKAGE_ROOT / "records_index.csv")
    parser.add_argument("--dataset-root", type=Path, help="Root containing original classified event-level waveform data")
    parser.add_argument("--npy", action="store_true", help="Explicitly load the original NPY instead")
    parser.add_argument("--trust-source", action="store_true", help="Trust the original pickle-bearing NPY source")
    parser.add_argument("--verify-all", action="store_true")
    parser.add_argument("--include-npy", action="store_true", help="Also hash original NPY files; does not unpickle them")
    parser.add_argument("--report", type=Path, help="Optional JSON verification report")
    args = parser.parse_args()
    if args.verify_all:
        if args.record or args.npy:
            parser.error("--verify-all cannot be combined with record or --npy")
        result = verify_all(args.index, args.dataset_root, args.include_npy)
    else:
        if not args.record or args.include_npy:
            parser.error("Supply a record, or use --verify-all [--include-npy]")
        value = load_event(args.record, index_path=args.index, dataset_root=args.dataset_root,
                           use_npy=args.npy, trust_source=args.trust_source)
        result = {**{name: value[name] for name in TEXT_FIELDS},
                  "sampling_rate": value["sampling_rate"], "n_samples": len(value["signal"]),
                  "format": "npy" if args.npy else "npz", "sha256_verified": True}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
