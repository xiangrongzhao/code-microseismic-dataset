# Open-pit mine microseismic dataset — companion code

Companion code for *A microseismic dataset for monitoring slope response to excavation disturbance in an open-pit mine*.

This repository provides the Python workflows named in the manuscript **Code Availability** section: waveform preprocessing, event detection and segmentation, event association, technical validation, and figure reproduction. It was tested with **Python 3.12**. Core dependencies are listed in `requirements.txt` (ObsPy, NumPy, pandas, scikit-learn, PyTorch, torchvision, Matplotlib).

[Processing guide](docs/workflows.md) · [中文导航](docs/README_zh.md) · [Figure index](figures/README.md)

## Repository layout (manuscript order)

| Stage | Directory | Role |
|---|---|---|
| 1. Preprocessing | `01_preprocessing/` | Band-pass filtering of continuous miniSEED |
| 2. Detection and segmentation | `02_event_detection/` | STA/LTA detection and window extraction |
| 3. Annotation template | `03_manual_annotation/` | Review template for expert class labels |
| 4. Event association | `event_association/` | Rebuild/verify the three catalogue CSV tables |
| 5. Waveform loading | `04_data_loading/` | Load classified `.npy` dictionaries |
| 6. Classification baselines | `05_classification/` | Whole-day split and training |
| 7. Technical validation | `06_validation/` | Integrity checks, metrics, validation plots |
| 8. Pickle-free loaders | `07_pickle_free_records/` | `load_event.py`, `build_npz_release.py` (NPZ archive on Figshare) |
| Figures | `figures/` | Entry point `figures/reproduce.py` and per-figure scripts |
| Bundled tables | `data/data logs/` | Catalogue CSVs, availability logs, `station-inventory.xml` |
| Saved experiments | `validation_results/` | Split, predictions and checkpoints used in the paper |
| Parameters | `parameters/` | Stage parameter JSON files |

Five labelled classes used throughout: **microseismic**, **blasting**, **small_landslide**, **mechanical_mining**, **transport_vehicles**.

## Install

~~~sh
python -m pip install -r requirements.txt
~~~

Optional figure/artwork checks:

~~~sh
python -m pip install -r figures/requirements-source.txt
~~~

## First run (bundled three-station example)

~~~sh
python 01_preprocessing/preprocess_raw.py --raw-dir examples/continuous --out-dir outputs/demo/filtered
python 02_event_detection/detect_events.py --raw-dir examples/continuous --out-dir outputs/demo/candidates
python 03_manual_annotation/review_candidates.py --candidates outputs/demo/candidates --out-dir outputs/demo/review
~~~

## Main workflows

~~~sh
# Event association (needs main dataset + pickle-free zip from Figshare)
python event_association/build_event_catalog.py --dataset "../dataset" --records-zip "../07_pickle_free_records.zip" --output outputs/catalogue
python event_association/verify_event_catalog.py outputs/catalogue --manifest validation_results/whole_day_split_manifest.csv

# Classification baselines
python 05_classification/split_manifest.py --data-root "../dataset/classified event-level waveform data" --split-policy whole-day-next --out outputs/whole_day_split_manifest.csv
python 05_classification/train_classification.py --data-root "../dataset/classified event-level waveform data" --split-manifest outputs/whole_day_split_manifest.csv --out-dir outputs/four_class --train-schema 4class --epochs 10 --batch-size 32 --seeds 42 --trust-source

# Technical validation / figures
python 06_validation/validate_dataset.py --events "../dataset/classified event-level waveform data" --trust-source --out outputs/event_integrity.json
python figures/reproduce.py --figure all
~~~

Pickle-free verification copies are deposited separately ([Figshare 34128987](https://doi.org/10.6084/m9.figshare.34128987)). After extracting them beside this repository:

~~~sh
python 07_pickle_free_records/load_event.py --verify-all
~~~

The loader checks SHA-256 checksums and loads `.npz` with `allow_pickle=False` by default. To read a verified original `.npy`, use `numpy.load(..., allow_pickle=True).item()`.

## Data links

- Main dataset: https://doi.org/10.6084/m9.figshare.34074750
- Pickle-free verification deposit: https://doi.org/10.6084/m9.figshare.34128987

## Verification

~~~sh
python verify_release.py
python 06_validation/check_saved_predictions.py
~~~

See [reproduction checks](docs/reproduction_checks.md).
