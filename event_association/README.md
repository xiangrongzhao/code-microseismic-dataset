# Event association

Rebuilds and verifies the three CSV tables described in the manuscript Data Records / Code Availability sections. The tables live in the dataset `data logs/` folder (and are mirrored under `data/data logs/` in this repository):

- `five_class_event_catalog.csv` — one row per catalogue event
- `waveform_to_event_id.csv` — one row per classified waveform file
- `blasting_station_count_distribution.csv` — blasting events by distinct station count

Event counts describe associations reconstructed from the retained vertical records. They do not measure how many blasts were fired at the mine. A station contributes once per event even when several windows are retained.

Five class labels: `microseismic`, `blasting`, `small_landslide`, `mechanical_mining`, `transport_vehicles`.

## Rebuild and verify

From this directory, set `DATASET` to the extracted main dataset and `NPZ_ZIP` to the pickle-free verification archive. Building and verifying the CSVs require only the Python standard library (plus the packages used elsewhere in this repository when auditing coverage).

~~~sh
python build_event_catalog.py --dataset "DATASET" --records-zip "NPZ_ZIP" --output "OUTPUT"
python verify_event_catalog.py "OUTPUT" --manifest "../validation_results/whole_day_split_manifest.csv"
python audit_blast_coverage.py --tables "OUTPUT" --dataset "DATASET"
~~~

To verify the bundled tables without rebuilding:

~~~sh
python verify_event_catalog.py "../data/data logs" --manifest "../validation_results/whole_day_split_manifest.csv"
~~~

## Grouping (blasting)

Sort by UTC trigger onset. Start a new group whenever the gap between consecutive onsets exceeds 3 s. The limit is applied to each adjacent pair, so the total span of a group may exceed 3 s. Event IDs use the first trigger time (`B` + `YYYYMMDD_HHMMSS`).

## Table fields (summary)

- **five_class_event_catalog.csv**: `event_id`, `event_label`, first/last trigger times, participating stations, per-station trigger JSON, unique-station and record counts, and per-array station counts.
- **waveform_to_event_id.csv**: record path, event ID, station ID, class label, trigger times; join to the catalogue on `event_id`.
- **blasting_station_count_distribution.csv**: counts of blasting events (and member files) for each distinct station number from 1 to 36.
