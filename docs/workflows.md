# Processing and validation workflows

All commands start at the repository root. Examples use `../dataset` for the extracted main Figshare dataset and `../07_pickle_free_records.zip` for the verification archive. Use fresh output directories. Stage parameters are listed in [parameters](../parameters).

These stages follow the manuscript **Code Availability** order: preprocessing → detection and segmentation → event association → technical validation → figures.

## 1–3. Continuous data, detection and labels

1. Run the [three-station example](../README.md#first-run-bundled-three-station-example) to inspect each intermediate result. For the full 1 June release, replace `examples/continuous` by `"../dataset/raw waveform data"`. Raw screening selects eligible DHZ records, checks sample counts and finite values, then applies a fourth-order zero-phase 1–20 Hz band-pass filter. Counts retain the instrument response.
2. Detection applies STA/LTA, groups overlapping trigger intervals from at least three distinct stations and extracts station windows. The nominal duration is 10 s; a data boundary can shorten a window.
3. The annotation script creates a review template. An expert assigns one of the five classes (`microseismic`, `blasting`, `small_landslide`, `mechanical_mining`, `transport_vehicles`) before reviewed windows are copied into class directories. Automated detection alone does not reproduce the archived expert labels.

~~~sh
python 01_preprocessing/preprocess_raw.py --raw-dir "../dataset/raw waveform data" --out-dir outputs/filtered
python 02_event_detection/detect_events.py --raw-dir "../dataset/raw waveform data" --out-dir outputs/candidates
python 03_manual_annotation/review_candidates.py --candidates outputs/candidates --out-dir outputs/review
python 06_validation/audit_raw_coverage.py --dataset ../dataset --out-dir outputs/coverage
python 06_validation/validate_dataset.py --events "../dataset/classified event-level waveform data" --trust-source --out outputs/event_integrity.json
~~~

## 4. Event catalogue (`event_association/`)

Rebuilds and verifies the three released tables in `data logs/`:

- `five_class_event_catalog.csv`
- `waveform_to_event_id.csv`
- `blasting_station_count_distribution.csv`

~~~sh
python event_association/build_event_catalog.py --dataset ../dataset --records-zip ../07_pickle_free_records.zip --output outputs/catalogue
python event_association/verify_event_catalog.py outputs/catalogue --manifest validation_results/whole_day_split_manifest.csv
python event_association/audit_blast_coverage.py --tables outputs/catalogue --dataset ../dataset
~~~

For a check without rebuilding, point `verify_event_catalog.py` at `"data/data logs"`. See [event_association/README.md](../event_association/README.md).

## 5–7. Classification and technical validation

The split is chronological by whole days within each native class. The four-class analysis merges microseismic and small-landslide labels into natural slope events.

~~~sh
python 05_classification/split_manifest.py --data-root "../dataset/classified event-level waveform data" --split-policy whole-day-next --out outputs/whole_day_split_manifest.csv
python 05_classification/train_classification.py --data-root "../dataset/classified event-level waveform data" --split-manifest outputs/whole_day_split_manifest.csv --out-dir outputs/four_class --train-schema 4class --epochs 10 --batch-size 32 --seeds 42 --trust-source
python 05_classification/train_classification.py --data-root "../dataset/classified event-level waveform data" --split-manifest outputs/whole_day_split_manifest.csv --out-dir outputs/five_class --train-schema 5class --epochs 8 --batch-size 32 --seeds 42 --trust-source
python 06_validation/validate_results.py --results outputs/four_class --split-manifest outputs/whole_day_split_manifest.csv --out outputs/four_class_verification.json
python 06_validation/plot_validation.py --results outputs/four_class --out outputs/figure07_from_training
python 06_validation/feature_statistics.py --results validation_results/four_class --out outputs/feature_statistics
~~~

See [RUN_EXPERIMENTS.txt](../05_classification/RUN_EXPERIMENTS.txt) and [validation_results/README.txt](../validation_results/README.txt).

## 8. Pickle-free loaders

Scripts live in `07_pickle_free_records/`. The NPZ archive itself is on Figshare ([34128987](https://doi.org/10.6084/m9.figshare.34128987)).

~~~sh
python 07_pickle_free_records/load_event.py --verify-all
~~~

## Figures

~~~sh
python figures/reproduce.py --figure 4
python figures/reproduce.py --figure all
~~~

Numerical outputs go to `outputs/figures/figureNN/`. Figures 3–7 have plotting workflows; Figures 1–2 use authored PowerPoint/artwork. See [figures/README.md](../figures/README.md).
