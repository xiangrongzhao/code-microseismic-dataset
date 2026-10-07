# code-microseismic-dataset


## Features

- Preprocess raw miniSEED waveforms with band-pass filtering and quality screening.
- Detect candidate events with STA/LTA and extract approximately 10-second event windows.
- Export candidate waveforms and label templates for manual annotation.
- Pick P-wave arrivals and reproduce a seven-station source-localization example using FMM and Levenberg-Marquardt (L-M).
- Train and evaluate five baseline classifiers on 17,622 labelled waveform records: CNN, Random Forest, Logistic Regression, MLP, and XGBoost. Both five-class and four-class tasks are supported.
- Run checks for data integrity, recording coverage, classification metrics, and saved predictions.

The five classes are microseismic events, blasting, small landslides, mechanical mining, and transport vehicles. The four-class task combines microseismic events and small landslides into natural slope events.

## Environment and installation

Use **Python 3.11 or newer**. The pinned versions below are the contents of `requirements.txt`; Python itself is installed separately. PyTorch is installed through the `torch` package.

```text
numpy==1.26.4
scipy==1.15.3
pandas==3.0.1
obspy==1.4.2
scikit-learn==1.6.1
xgboost==3.0.2
torch==2.7.1
torchvision==0.22.1
matplotlib==3.10.3
contourpy==1.3.2
scikit-fmm==2025.6.23
joblib==1.6.0
setuptools==75.8.0
pyproj==3.7.1
```

Install from the supplied `code-dataset` directory:

```shell
pip install -r requirements.txt
```

A CUDA-capable GPU can speed up CNN training. The first CNN run downloads torchvision's pretrained ResNet-18 weights unless they are already cached.

## Quick start

### 1. Download and extract the dataset

Download the dataset from Figshare: https://doi.org/10.6084/m9.figshare.34074750

```text
project/
|-- code-dataset/
|-- dataset/
|   |-- raw waveform data/
|   |-- classified event-level waveform data/
|   |-- station metadata/
|   `-- data logs/
`-- experiment-results/                 # Created by the scripts
```

The classified waveform directory contains 17,622 `.npy` files: 1,401 microseismic, 6,512 blasting, 148 small-landslide, 7,690 mechanical-mining, and 1,871 transport-vehicle records. The raw example contains 15,360 miniSEED files for 1 June 2025.

### 2. Set your local paths

Run all commands from `code-dataset`. Replace `../dataset` with your extracted dataset path and `../experiment-results` with your output path. Paths are supplied through command-line arguments, so no configuration-file edit is required. Keep generated files outside the code and source-data directories.

The original `.npy` dictionaries use pickle. Use `--trust-source` when reading the trusted dataset download with the existing workflows. The `09_pickle_free_records` folder contains all 17,622 classified records as pickle-free `.npz` files, `records_index.csv` with SHA-256 hashes of both formats, the indexed loader `load_event.py`, the conversion script `build_npz_release.py`, the conversion report `conversion_report.json` and `README.txt`. It serves the verification of the dataset in the same way as this code package and is not part of the dataset release; because of its data volume it is not stored in the GitHub repository but deposited separately on Figshare: https://doi.org/10.6084/m9.figshare.34128987. Extract it inside `code-dataset` to run the command below. It works independently of the original NPY files for NPZ loading. See `09_pickle_free_records/README.txt` for loading and full-release verification.

```shell
python 09_pickle_free_records/load_event.py --verify-all
```

### 3. Run the workflows

Preprocess raw files, detect candidates, and export a manual-review template:

```shell
python 01_preprocessing/preprocess_raw.py --raw-dir "../dataset/raw waveform data" --out-dir ../experiment-results/preprocessed
python 02_event_detection/detect_events.py --raw-dir "../dataset/raw waveform data" --out-dir ../experiment-results/candidates
python 03_manual_annotation/review_candidates.py --candidates ../experiment-results/candidates --out-dir ../experiment-results/manual_review
```

Detection reads the raw miniSEED files and applies preprocessing internally. Fill in `annotation_template.json` to review newly detected candidates. Classification below uses the released, already labelled event records.

Create the shared chronological split, then run both classification tasks:

```shell
python 07_classification/split_manifest.py --data-root "../dataset/classified event-level waveform data" --split-policy whole-day-next --out ../experiment-results/whole_day_split_manifest.csv
python 07_classification/train_classification.py --data-root "../dataset/classified event-level waveform data" --split-manifest ../experiment-results/whole_day_split_manifest.csv --out-dir ../experiment-results/five_class --train-schema 5class --epochs 8 --batch-size 32 --seeds 42 --trust-source
python 07_classification/train_classification.py --data-root "../dataset/classified event-level waveform data" --split-manifest ../experiment-results/whole_day_split_manifest.csv --out-dir ../experiment-results/four_class --train-schema 4class --epochs 10 --batch-size 32 --seeds 42 --trust-source
```

The supplied archive produces 11,437 training, 4,165 validation, and 2,020 test records. Both tasks use the same file assignments. Date separation is applied within each original class. See [RUN_EXPERIMENTS.txt](07_classification/RUN_EXPERIMENTS.txt) for evaluation and figure commands.

Regenerate arrivals from the seven bundled waveforms and reproduce the localization example:

```shell
python 05_velocity_localization/make_inputs1.py --out-dir ../experiment-results/localization_inputs
python 05_velocity_localization/run_localization_example.py --inputs ../experiment-results/localization_inputs --out-dir ../experiment-results/localization
```

The localization example includes its waveforms, station table, velocity model (`XJD2_updated.txt`) and the pick set `05_velocity_localization/code_inputs1` (`stas_xyz.npy`, `tobs_p.npy`, `pick_report.csv`, `input_provenance.json`), which `make_inputs1.py` rebuilds from the bundled waveforms. To reproduce the reported solutions directly from the bundled picks:

```shell
python 05_velocity_localization/run_localization_example.py --inputs 05_velocity_localization/code_inputs1 --out-dir ../experiment-results/localization
```

See [the localization guide](05_velocity_localization/README.txt) for expected values and verification.

Parameter files for every processing stage are in `parameters/` (one JSON file per stage, from raw-data screening to classification).

## Validation

The `validation_results` directory contains the split manifest, model checkpoints, predictions and metric files that reproduce the tables and figures of the Technical Validation section, together with the localization-example results. See `validation_results/README.txt` for its contents and for the regenerable intermediate files that are not included.

Check the event files, recording coverage, and saved results:

```shell
python 08_validation/validate_dataset.py --events "../dataset/classified event-level waveform data" --out ../experiment-results/dataset_validation.json --trust-source
python 08_validation/audit_raw_coverage.py --dataset ../dataset --out-dir ../experiment-results/coverage
python 08_validation/validate_results.py --results ../experiment-results/five_class --out ../experiment-results/five_class_verification.json
python 08_validation/validate_results.py --results ../experiment-results/four_class --out ../experiment-results/four_class_verification.json
```

## Directory structure

```text
code-microseismic-dataset/
|-- 01_preprocessing/           # Raw-waveform filtering and quality screening
|-- 02_event_detection/         # STA/LTA detection and event-window extraction
|-- 03_manual_annotation/       # Candidate review and annotation templates
|-- 04_arrival_picking/         # General P-wave arrival picking
|-- 05_velocity_localization/   # Velocity model, code_inputs1 and localization example
|-- 06_data_loading/            # NumPy event loading and schema checks
|-- 07_classification/          # Baseline training and evaluation
|-- 08_validation/              # Integrity, coverage, metrics, and figures
|-- parameters/                 # Parameter files for every processing stage
|-- validation_results/         # Split manifest, checkpoints, predictions and metrics
|-- README.md                  # Project guide
|-- README.txt                 # Plain-text copy of this guide
|-- LICENSE                    # CC BY 4.0 license text
`-- requirements.txt           # Python dependencies
```

`09_pickle_free_records/` (17,622 NPZ records, SHA-256 index, indexed loader, conversion script and report) is deposited separately on Figshare (https://doi.org/10.6084/m9.figshare.34128987) and can be extracted into this directory.

## Outputs

| Workflow | Main outputs |
|---|---|
| Preprocessing | Filtered `.npz` waveforms and `preprocessing.json` |
| Detection and review | Candidate `.npy` records, `detection_summary.json`, and `annotation_template.json` |
| Classification | Split assignments, model checkpoints, predictions, `multimodel_metrics.csv`, and `summary.json` |
| Localization | Rebuilt arrival arrays, `pick_report.csv`, `localization_summary.json`, and Figure 5c/d panels |
| Validation | Integrity reports, coverage tables, and recalculated classification metrics |

Classification reports include accuracy, balanced accuracy, macro-F1, per-class scores, and confusion matrices. Use a fresh classification output directory when changing the data, split, or settings. In the localization example, the released-model L-M iteration stops when damping becomes too large; its reproducible result does not demonstrate convergence or independently measured location accuracy.

## License and citation

This project is released under [Creative Commons Attribution 4.0 International (CC BY 4.0)], matching the dataset license. You may copy, redistribute, modify, and use the original material commercially, provided you credit the authors, link to the license, and identify changes. See [LICENSE](LICENSE).


