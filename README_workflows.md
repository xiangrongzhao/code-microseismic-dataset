# Workflow quick reference

Detailed commands are in [docs/workflows.md](docs/workflows.md). This file mirrors the manuscript **Code Availability** sequence.

## Directory map

~~~
01_preprocessing/          # continuous miniSEED filtering
02_event_detection/        # STA/LTA detection and segmentation
03_manual_annotation/      # expert-review template
event_association/         # rebuild/verify the three catalogue CSVs
04_data_loading/           # classified NPY loading
05_classification/         # whole-day split and baselines
06_validation/             # integrity, metrics, validation plots
07_pickle_free_records/    # load_event.py / build_npz_release.py
figures/                   # figures/reproduce.py
data/data logs/            # bundled catalogue and availability tables
validation_results/        # archived experiment outputs
parameters/                # stage JSON settings
~~~

## Pickle-free verification archive

The original `.npy` dictionaries use pickle. Use `--trust-source` when reading the trusted dataset download. The pickle-free `.npz` copies, `records_index.csv`, `load_event.py`, `build_npz_release.py`, `conversion_report.json` and `README.txt` are deposited on Figshare: https://doi.org/10.6084/m9.figshare.34128987. Extract that archive beside this repository (or into `07_pickle_free_records/` as documented there), then run:

~~~sh
python 07_pickle_free_records/load_event.py --verify-all
~~~

## Classification

~~~sh
python 05_classification/split_manifest.py --data-root "../dataset/classified event-level waveform data" --split-policy whole-day-next --out outputs/whole_day_split_manifest.csv
python 05_classification/train_classification.py --data-root "../dataset/classified event-level waveform data" --split-manifest outputs/whole_day_split_manifest.csv --out-dir outputs/five_class --train-schema 5class --epochs 8 --batch-size 32 --seeds 42 --trust-source
python 05_classification/train_classification.py --data-root "../dataset/classified event-level waveform data" --split-manifest outputs/whole_day_split_manifest.csv --out-dir outputs/four_class --train-schema 4class --epochs 10 --batch-size 32 --seeds 42 --trust-source
~~~

See [05_classification/RUN_EXPERIMENTS.txt](05_classification/RUN_EXPERIMENTS.txt).

## Validation

~~~sh
python 06_validation/validate_dataset.py --events "../dataset/classified event-level waveform data" --out outputs/dataset_validation.json --trust-source
python 06_validation/audit_raw_coverage.py --dataset ../dataset --out-dir outputs/coverage
python 06_validation/validate_results.py --results outputs/four_class --out outputs/four_class_verification.json
~~~
