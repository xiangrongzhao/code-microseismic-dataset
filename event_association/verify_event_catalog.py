"""Verify keys, membership, station counts, the Fig. 8 sample and the entire distribution."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from catalogue_paths import table_path


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify(folder, manifest=None):
    events = read(table_path(folder, "five_class_event_catalog.csv"))
    records = read(table_path(folder, "waveform_to_event_id.csv"))
    dist = read(table_path(folder, "blasting_station_count_distribution.csv"))
    assert len(records)==len({r["relative_path"] for r in records})==len({r["record_id"] for r in records})==17622
    assert Counter(r["record_label"] for r in records)==dict(blasting=6512, microseismic=1401, small_landslide=148, mechanical_mining=7690, transport_vehicles=1871)
    byid = defaultdict(list)
    for r in records:
        assert r["event_id"] and r["trigger_on_utc"] <= r["trigger_off_utc"]
        byid[r["event_id"]].append(r)
    assert set(byid)=={e["event_id"] for e in events} and len(byid)==len(events)
    counts, file_counts = Counter(), Counter()
    for event in events:
        rs = byid[event["event_id"]]
        times = defaultdict(list)
        for r in rs:
            assert r["record_label"]==event["event_label"]
            times[r["station_id"]].append(r["trigger_on_utc"])
        assert {s:sorted(v) for s,v in times.items()}==json.loads(event["station_triggers_utc"])
        assert sorted(times)==event["station_ids"].split(";") and len(times)==int(event["n_unique_stations"])
        assert len(rs)==int(event["n_records"])
        assert min(r["trigger_on_utc"] for r in rs)==event["first_trigger_utc"]
        assert max(r["trigger_on_utc"] for r in rs)==event["last_trigger_utc"]
        for name,key in (("southern_highwall","n_southern_highwall"),("internal_dump","n_internal_dump")):
            assert len({r["station_id"] for r in rs if r["station_region"]==name})==int(event[key])
        if event["event_label"]=="blasting":
            counts[len(times)]+=1
            file_counts[len(times)]+=len(rs)
    blasts = sorted((r for r in records if r["record_label"]=="blasting"),key=lambda r:r["trigger_on_utc"])
    for a,b in zip(blasts, blasts[1:]):
        delta = datetime.fromisoformat(b["trigger_on_utc"].rstrip("Z"))-datetime.fromisoformat(a["trigger_on_utc"].rstrip("Z"))
        assert (delta<=timedelta(seconds=3))==(a["event_id"]==b["event_id"])
    assert len(dist)==36
    for k,row in enumerate(dist,1):
        assert k==int(row["n_unique_stations"])
        assert int(row["n_blasting_events"])==counts[k] and int(row["n_blasting_records"])==file_counts[k]
        assert abs(float(row["percent_of_catalogued_blasts"])-100*counts[k]/sum(counts.values()))<.000051
    if manifest:
        splits={r["relative_path"]:r["split"] for r in read(manifest)}
        crossing=[k for k,rs in byid.items() if len({splits[r["relative_path"].split("/",1)[1]] for r in rs})>1]
        print("Groups crossing archived split:",len(crossing))
    print("Verified records",len(records),"events",len(events),"blasting events",sum(counts.values()),"blasting records",sum(file_counts.values()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("tables",type=Path)
    p.add_argument("--manifest",type=Path)
    a=p.parse_args()
    verify(a.tables,a.manifest)
