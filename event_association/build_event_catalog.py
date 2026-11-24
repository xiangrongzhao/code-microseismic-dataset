"""Build the five-class event directory from the released waveform inventory."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

CLASSES = ("blasting", "microseismic", "small_landslide", "mechanical_mining", "transport_vehicles")
HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utc(value):
    return datetime.fromisoformat(value.rstrip("Z"))


def sort_key(row):
    return row["on"], row["record_id"]


def inventory(dataset, archive):
    with ZipFile(archive) as files:
        index = next(name for name in files.namelist() if name.endswith("/records_index.csv"))
        with files.open(index) as stream:
            rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
    paths = {p.relative_to(dataset).as_posix() for p in (dataset/"classified event-level waveform data").rglob("*.npy")}
    if paths != {r["npy_path"] for r in rows} or len(rows) != len(paths):
        raise ValueError("Index and released waveform inventory differ")
    stations = {r["station_id"]: r["array"] for r in read_csv(HERE/"figure8"/"stations.csv")}
    for r in rows:
        if hashlib.sha256((dataset/r["npy_path"]).read_bytes()).hexdigest() != r["npy_sha256"]:
            raise ValueError("Checksum mismatch: " + r["record_id"])
        r["on"], r["off"] = utc(r["event_start"]), utc(r["event_end"])
        r["region"] = stations[r["sensor_id"]]
        if r["off"] < r["on"] or r["class"] not in CLASSES:
            raise ValueError(r["record_id"])
    return sorted(rows, key=sort_key)


def blast_groups(rows):
    groups = []
    for row in rows:
        if not groups or row["on"] - groups[-1][-1]["on"] > timedelta(seconds=3):
            groups.append([])
        groups[-1].append(row)
    return groups


def other_groups(rows, documents):
    # Preserve the matched June rerun groups.
    lookup = {r["record_id"]: r for r in rows}
    groups = []
    used = set()
    for doc in documents:
        ids = doc["record_ids"]
        if used.intersection(ids):
            raise ValueError("Duplicate documented membership")
        groups.append([lookup[i] for i in ids])
        used.update(ids)
    # Use the detector's earliest-reference interval and chronological tie breaking.
    # The second pass handles the station records remaining after archival selection.
    for minimum in (3, 2):
        remaining = [r for r in rows if r["record_id"] not in used]
        for i, anchor in enumerate(remaining):
            if anchor["record_id"] in used:
                continue
            members = [anchor]
            j = i + 1
            while j < len(remaining) and remaining[j]["on"] <= anchor["off"]:
                row = remaining[j]
                if row["record_id"] not in used:
                    members.append(row)
                j += 1
            if len({r["sensor_id"] for r in members}) >= minimum:
                groups.append(members)
                used.update(r["record_id"] for r in members)
    groups.extend([r] for r in rows if r["record_id"] not in used)
    return groups


def make_tables(rows):
    documents = json.loads((HERE/"association_sources.json").read_text("utf-8"))["groups"]
    events, records = [], []
    for label in CLASSES:
        subset = [r for r in rows if r["class"] == label]
        groups = blast_groups(subset) if label == "blasting" else other_groups(subset, [d for d in documents if d["event_label"] == label])
        for group in groups:
            group.sort(key=sort_key)
            first = group[0]["on"]
            if label == "blasting":
                eid = first.strftime("B%Y%m%d_%H%M%S")
            else:
                key = "\n".join(sorted(r["record_id"] for r in group))
                eid = label.upper() + "-" + hashlib.sha256(key.encode()).hexdigest()[:16]
            times = defaultdict(list)
            arrays = {}
            for r in group:
                station = r["sensor_id"]
                onset = r["on"].isoformat(timespec="microseconds") + "Z"
                times[station].append(onset)
                arrays[station] = r["region"]
                records.append(dict(record_id=r["record_id"], relative_path=r["npy_path"], event_id=eid,
                                    station_id=station, trigger_on_utc=onset, trigger_off_utc=r["off"].isoformat(timespec="microseconds")+"Z",
                                    record_label=label, station_region=r["region"]))
            stations = sorted(times)
            regions = Counter(arrays.values())
            events.append(dict(event_id=eid, event_label=label, first_trigger_utc=times[group[0]["sensor_id"]][0],
                               last_trigger_utc=group[-1]["on"].isoformat(timespec="microseconds")+"Z",
                               n_unique_stations=len(stations), n_records=len(group), station_ids=";".join(stations),
                               station_triggers_utc=json.dumps(dict(sorted(times.items())), separators=(",", ":")),
                               n_southern_highwall=regions["southern_highwall"], n_internal_dump=regions["internal_dump"]))
    events.sort(key=lambda r: (r["first_trigger_utc"],r["event_id"]))
    records.sort(key=lambda r:r["record_id"])
    if len({r["event_id"] for r in events}) != len(events) or len({r["record_id"] for r in records}) != len(rows) or len(rows) != len(records):
        raise ValueError("IDs or memberships are not unique")
    blasts = [e for e in events if e["event_label"] == "blasting"]
    distribution = []
    for k in range(1,37):
        members = [e for e in blasts if e["n_unique_stations"] == k]
        distribution.append(dict(n_unique_stations=k, n_blasting_events=len(members),
                                 n_blasting_records=sum(e["n_records"] for e in members),
                                 percent_of_catalogued_blasts=round(100*len(members)/len(blasts),4)))
    return events, records, distribution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--records-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = inventory(args.dataset, args.records_zip)
    result = make_tables(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in zip(("five_class_event_catalog.csv","waveform_to_event_id.csv","blasting_station_count_distribution.csv"),result):
        write_csv(args.output/name, table)
    print("Events by class", dict(Counter(e["event_label"] for e in result[0])))
    print("Records by class", dict(Counter(r["record_label"] for r in result[1])))


if __name__ == "__main__":
    main()
