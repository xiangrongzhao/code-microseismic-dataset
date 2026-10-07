VALIDATION RESULTS

These files reproduce the tables and figures of the Technical Validation
section: 17,622 labelled records, chronological whole-day split of 11,437
training, 4,165 validation and 2,020 test records, seed 42.

whole_day_split_manifest.csv   File-by-file split manifest used by both tasks.
whole_day_split_summary.json   Per-class split counts and date ranges.
five_class_verification.json   Independent recomputation of the five-class metrics.
four_class_verification.json   Independent recomputation of the four-class metrics.
five_class/, four_class/       Run configuration, model checkpoints
                               (cnn_checkpoint.pt, model_*.joblib), test and
                               validation predictions (proba_*.npy,
                               validation_proba_*.npy), confusion matrices,
                               metric files (multimodel_metrics.csv,
                               validation_metrics.csv, summary.json), training
                               logs, feature statistics and the t-SNE projection.
                               The ensemble is the mean of all five models
                               (five classes) or of CNN and Random Forest (four
                               classes).
localization/                  Results of the localization example
                               (localization_summary.json, Figure 5c/d panels).

Paths in run_config.json and dataset_fingerprint.json are relative to the
code-dataset directory and the dataset location used in README.txt.

To keep every file below 100 MB, the following regenerable intermediate files
are not included: spectrograms.npy, features.npy, features_finetuned.npy,
features_imagenet.npy and the two ImageNet-comparison models
ablation_finetuned.joblib and ablation_imagenet_frozen.joblib. Their
predictions and metrics (ablation_*_proba.npy, ablation_*_metrics.json,
imagenet_vs_finetuned_comparison.csv) are included. Running
07_classification/train_classification.py with the commands in README.txt and
this split manifest regenerates the omitted files; 08_validation/validate_results.py
requires them.
