"""Check the published validation and test metrics from stored probabilities without retraining."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


def check(run):
    summary = json.loads((run / "summary.json").read_text(encoding="utf8"))
    labels = np.arange(len(summary["classes"]))
    for prefix, phase, table in [("", "test", "multimodel_metrics.csv"), ("validation_", "validation", "validation_metrics.csv")]:
        y = np.load(run / (phase + "_labels.npy"), allow_pickle=False)
        with (run / table).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            model = row["model"]
            proba = np.load(run / (prefix + "proba_" + model + ".npy"), allow_pickle=False)
            prediction = proba.argmax(axis=1)
            if proba.shape != (len(y), len(labels)):
                raise ValueError(f"Invalid probability dimensions for {phase}/{model}")
            precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, labels=labels, zero_division=0)
            values = [accuracy_score(y, prediction), precision.mean(), recall.mean(), f1.mean(), balanced_accuracy_score(y, prediction)]
            fields = ["overall_accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy"]
            np.testing.assert_allclose(values, [float(row[field]) for field in fields], rtol=0, atol=1e-12)
            saved = np.load(run / (prefix + "confusion_" + model + ".npy"), allow_pickle=False)
            np.testing.assert_array_equal(confusion_matrix(y, prediction, labels=labels), saved)
        members = [np.load(run / (prefix + "proba_" + name + ".npy"), allow_pickle=False) for name in summary["ensemble_members"]]
        ensemble = np.load(run / (prefix + "proba_Ensemble.npy"), allow_pickle=False)
        np.testing.assert_allclose(np.mean(members, axis=0, dtype=np.float64), ensemble, rtol=1e-6, atol=1e-7)
        print(f"{run.name} {phase}: verified {len(y)} labels and {len(rows)} result rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "validation_results"
    for run in args.results or [root / "four_class", root / "five_class"]:
        check(run)
