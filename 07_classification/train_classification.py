"""Classification demonstration for the curated vertical-component archive.
Five labels and the merged four-label schema share a verified within-class
chronological file partition. All model/scaler fitting uses training files.
Original implementation adapted to portable paths, isolated result directories,
explicit ensemble membership, data fingerprints and saved model checkpoints.
Only load the NumPy archive if its source is trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import copy
import json
import math
import os
import time
import joblib
import random
import platform
import importlib.metadata
from pathlib import Path
from split_manifest import load_manifest, assert_cache_signature

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from scipy.signal import spectrogram as scipy_spectrogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
CLASS_DIRS = [
    "microseismic",
    "blasting",
    "small_landslide",
    "mechanical_mining",
    "transport_vehicles",
]
CLASS_KEYS = [
    "microseismic",
    "blasting",
    "small_landslide",
    "mechanical_mining",
    "transport_vehicles",
]
# 4-class schema: merge "small_landslide" (idx=2) into "microseismic" (idx=0).
# The remaining three classes keep their 5-class semantics.  We re-map as:
#    5-class idx   0 1 2 3 4  ->  4-class idx   0 1 0 2 3
CLASS_KEYS_4 = [
    "natural_slope_events",  # microseismic + small_landslide
    "blasting",
    "mechanical_mining",
    "transport_vehicles",
]
MAP_5_TO_4 = np.array([0, 1, 0, 2, 3], dtype=np.int64)

SEED_DEFAULTS = (42,)
SPEC_SHAPE = (128, 128)
SAMPLE_LENGTH = 2500
DATA_DIR = Path(__file__).resolve().parent / "data"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setup_cpu_threads() -> None:
    """Force torch to use all physical CPU cores when CUDA is unavailable."""
    n = os.cpu_count() or 4
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 4))
    except RuntimeError:
        pass  # already initialised, benign


# ---------------------------------------------------------------------------
# stage 0: signals -> spectrograms (only runs if cache missing; mirrors
# train_classification_fulldata.py so both pipelines share identical splits)
# ---------------------------------------------------------------------------
def load_signal(path: str) -> np.ndarray | None:
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            item = arr.item()
            if isinstance(item, dict) and "signal" in item:
                sig = np.asarray(item["signal"], dtype=np.float32)
            else:
                sig = np.asarray(item, dtype=np.float32).ravel()
        except Exception:
            return None
    else:
        sig = np.asarray(arr, dtype=np.float32).ravel()
    if sig.size == 0 or not np.isfinite(sig).all():
        return None
    if sig.size > SAMPLE_LENGTH:
        start = (sig.size - SAMPLE_LENGTH) // 2
        sig = sig[start:start + SAMPLE_LENGTH]
    elif sig.size < SAMPLE_LENGTH:
        pad = SAMPLE_LENGTH - sig.size
        sig = np.pad(sig, (pad // 2, pad - pad // 2))
    std = float(np.std(sig))
    if std > 0:
        sig = sig / std
    return sig


def compute_spectrogram(sig: np.ndarray) -> np.ndarray:
    _, _, Sxx = scipy_spectrogram(
        sig, fs=250, nperseg=254, noverlap=234, nfft=254, scaling="spectrum",
    )
    Sxx = np.log1p(Sxx.astype(np.float32))
    zy = SPEC_SHAPE[0] / Sxx.shape[0]
    zx = SPEC_SHAPE[1] / Sxx.shape[1]
    spec = zoom(Sxx, (zy, zx), order=1)
    spec = spec[:SPEC_SHAPE[0], :SPEC_SHAPE[1]]
    if spec.shape != SPEC_SHAPE:
        padded = np.zeros(SPEC_SHAPE, dtype=np.float32)
        padded[:spec.shape[0], :spec.shape[1]] = spec
        spec = padded
    mu, sd = spec.mean(), spec.std() + 1e-6
    spec = (spec - mu) / sd
    return spec.astype(np.float16)


def ensure_cache(data_root: Path, force: bool, fixed_frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits_path = DATA_DIR / "splits.csv"
    if splits_path.exists() and not force:
        splits_df = pd.read_csv(splits_path)
    else:
        splits_df = fixed_frame.copy()
        splits_df.to_csv(splits_path, index=False, encoding="utf-8")
        log(f"wrote {splits_path}")

    spec_path = DATA_DIR / "spectrograms.npy"
    valid_path = DATA_DIR / "valid_mask.npy"
    if spec_path.exists() and valid_path.exists() and not force:
        specs = np.load(spec_path, mmap_mode="r")
        valid = np.load(valid_path)
        log(f"spectrogram cache hit: {spec_path}  valid={int(valid.sum())}")
        return splits_df, specs, valid

    log(f"computing {len(splits_df)} spectrograms...")
    specs = np.zeros((len(splits_df), *SPEC_SHAPE), dtype=np.float16)
    valid = np.ones(len(splits_df), dtype=bool)
    t0 = time.time()
    for i, row in enumerate(splits_df.itertuples(index=False)):
        sig = load_signal(row.path)
        if sig is None:
            raise ValueError(f'Cannot load finite waveform: {row.path}')
        try:
            specs[i] = compute_spectrogram(sig)
        except Exception as e:
            raise ValueError(f'Cannot compute spectrogram: {row.path}') from e
        if (i + 1) % 2000 == 0:
            dt = time.time() - t0
            log(f"  {i + 1}/{len(splits_df)}  valid={int(valid[:i + 1].sum())}  "
                f"({dt:.1f}s)")
    np.save(spec_path, specs)
    np.save(valid_path, valid)
    log(f"wrote {spec_path}  shape={specs.shape}")
    return splits_df, specs, valid


# ---------------------------------------------------------------------------
# stage 1: ImageNet-frozen features (Track-B baseline, kept for ablation)
# ---------------------------------------------------------------------------
def export_imagenet_features(specs: np.ndarray, valid: np.ndarray,
                              device: torch.device, force: bool) -> np.ndarray:
    out = DATA_DIR / "features_imagenet.npy"
    if out.exists() and not force:
        log(f"features_imagenet cache hit: {out}")
        return np.load(out)
    log("building frozen ImageNet ResNet18 for baseline features...")
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)

    N = len(specs)
    feats = np.zeros((N, 512), dtype=np.float32)
    batch_size = 32
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            chunk = specs[start:start + batch_size].astype(np.float32)
            t = torch.from_numpy(chunk).unsqueeze(1).to(device)
            t = F.interpolate(t, size=(224, 224), mode="bilinear",
                              align_corners=False)
            t = t.repeat(1, 3, 1, 1)
            t = (t - mean) / std_t
            feats[start:start + len(chunk)] = model(t).detach().cpu().numpy()
            if (start // batch_size) % 50 == 0:
                dt = time.time() - t0
                log(f"  imagenet feats {start + len(chunk)}/{N}  "
                    f"elapsed={dt:.0f}s")
    feats[~valid] = 0.0
    np.save(out, feats)
    log(f"wrote {out}")
    return feats


# ---------------------------------------------------------------------------
# stage 2: fine-tuning core
# ---------------------------------------------------------------------------
class SpecDataset(Dataset):
    """Serve 1-channel 128x128 spectrograms with optional augmentation."""
    def __init__(self, specs: np.ndarray, labels: np.ndarray,
                 indices: np.ndarray, augment: bool,
                 freq_mask: int = 16, time_mask: int = 16,
                 amp_jitter: tuple[float, float] = (0.8, 1.25), seed: int = 42):
        self.specs = specs
        self.labels = labels
        self.indices = indices
        self.augment = augment
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.amp_jitter = amp_jitter
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        real = int(self.indices[i])
        spec = self.specs[real].astype(np.float32)
        if self.augment:
            # amplitude jitter (equivalent to small gain variation)
            lo, hi = self.amp_jitter
            spec = spec * float(self._rng.uniform(lo, hi))
            # SpecAugment: frequency masking
            for _ in range(2):
                f = int(self._rng.integers(0, self.freq_mask + 1))
                f0 = int(self._rng.integers(0, max(1, SPEC_SHAPE[0] - f)))
                spec[f0:f0 + f, :] = 0.0
            # SpecAugment: time masking
            for _ in range(2):
                t = int(self._rng.integers(0, self.time_mask + 1))
                t0 = int(self._rng.integers(0, max(1, SPEC_SHAPE[1] - t)))
                spec[:, t0:t0 + t] = 0.0
        x = torch.from_numpy(spec).unsqueeze(0)  # (1, 128, 128)
        y = int(self.labels[real])
        return x, y


def build_resnet18_1ch(num_classes: int,
                      imagenet: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if imagenet else None
    model = models.resnet18(weights=weights)
    # Replace first conv with 1-channel variant; average the original
    # 3-channel weights so ImageNet priors are preserved.
    old = model.conv1
    new = nn.Conv2d(
        in_channels=1,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    with torch.no_grad():
        if imagenet:
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced loss with an optional focal modulator.

    ``gamma=0`` recovers plain weighted cross-entropy (the default for the
    released pipeline: it converges much faster on CPU than focal with
    ``gamma=2`` and is numerically more stable when the WeightedRandomSampler
    also balances the mini-batches).  ``weight_mode`` selects how class
    counts are mapped to loss weights:

    - ``"effective"`` (Cui et al. 2019): ``(1-beta)/(1-beta**N)`` normalised
      to mean 1.
    - ``"log"``: ``w_c = log(N_max / N_c) + 1``, a softer correction that
      pairs well with WeightedRandomSampler and avoids the 100x weight
      blow-up that makes 155-sample classes destabilise training.
    - ``"sqrt"``: ``w_c = sqrt(N_max / N_c)`` - middle ground.
    - ``"none"``: uniform weights.
    """
    def __init__(self, per_class_count: np.ndarray,
                 beta: float = 0.9999, gamma: float = 0.0,
                 weight_mode: str = "log"):
        super().__init__()
        counts = per_class_count.astype(np.float64)
        if weight_mode == "effective":
            effective = 1.0 - np.power(beta, counts)
            w = (1.0 - beta) / np.maximum(effective, 1e-12)
        elif weight_mode == "log":
            w = np.log(counts.max() / np.maximum(counts, 1)) + 1.0
        elif weight_mode == "sqrt":
            w = np.sqrt(counts.max() / np.maximum(counts, 1))
        elif weight_mode == "none":
            w = np.ones_like(counts)
        else:
            raise ValueError(f"unknown weight_mode={weight_mode}")
        w = w / w.sum() * len(w)  # normalise to mean 1
        self.register_buffer("cb_weights",
                             torch.tensor(w, dtype=torch.float32))
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.cb_weights,
                             reduction="none")
        if self.gamma > 0:
            pt = torch.exp(-ce)
            loss = ((1 - pt) ** self.gamma) * ce
        else:
            loss = ce
        return loss.mean()


def layerwise_param_groups(model: nn.Module,
                           lr_head: float, lr_backbone: float,
                           weight_decay: float) -> list[dict]:
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": lr_backbone,
         "weight_decay": weight_decay, "name": "backbone"},
        {"params": head_params, "lr": lr_head,
         "weight_decay": weight_decay, "name": "head"},
    ]


def train_one_seed(specs: np.ndarray, valid: np.ndarray,
                   splits_df: pd.DataFrame,
                   device: torch.device,
                   seed: int,
                   epochs: int, batch_size: int,
                   lr_head: float, lr_backbone: float,
                   weight_decay: float,
                   patience: int,
                   weight_mode: str = "log",
                   focal_gamma: float = 0.0,
                   use_sampler: bool = True,
                   sampler_power: float = 1.0 / 3.0,
                   class_keys: list[str] | None = None,
                   label_mapper: np.ndarray | None = None,
                   mixup_alpha: float = 0.0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    active_keys = class_keys if class_keys is not None else CLASS_KEYS
    num_classes = len(active_keys)

    y_raw = splits_df["class_idx"].to_numpy(dtype=np.int64)
    y_all = label_mapper[y_raw] if label_mapper is not None else y_raw
    is_valid = valid.astype(bool)
    split = splits_df["split"].to_numpy()

    train_idx = np.flatnonzero((split == "train") & is_valid)
    val_idx = np.flatnonzero((split == "val") & is_valid)
    test_idx = np.flatnonzero((split == "test") & is_valid)

    per_class = np.bincount(y_all[train_idx], minlength=num_classes)
    log(f"  [seed={seed}] train per-class ({num_classes}-class): {per_class.tolist()}")

    if use_sampler:
        # WeightedRandomSampler with tempered power (default cube-root).
        w_class = 1.0 / np.power(per_class.astype(np.float64) + 1e-6,
                                 sampler_power)
        w_class = w_class / w_class.sum() * len(w_class)
        sample_w = w_class[y_all[train_idx]]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(train_idx), replacement=True,
            generator=torch.Generator().manual_seed(seed + 17),
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_ds = SpecDataset(specs, y_all, train_idx, augment=True, seed=seed)
    val_ds = SpecDataset(specs, y_all, val_idx, augment=False)
    test_ds = SpecDataset(specs, y_all, test_idx, augment=False)

    num_workers = 0  # Windows-safe default
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin,
                              generator=torch.Generator().manual_seed(seed + 23),
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

    model = build_resnet18_1ch(num_classes, imagenet=True).to(device)
    loss_fn = ClassBalancedFocalLoss(
        per_class, gamma=focal_gamma, weight_mode=weight_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        layerwise_param_groups(model, lr_head, lr_backbone, weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    stale = 0
    history = []

    rng_mixup = np.random.default_rng(seed + 101)
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if mixup_alpha > 0 and rng_mixup.random() < 0.5:
                # Beta-distributed lambda, mix with a random permutation.
                lam = float(rng_mixup.beta(mixup_alpha, mixup_alpha))
                lam = max(lam, 1.0 - lam)  # keep lam >= 0.5 so majority label is semantically ``yb``
                perm = torch.randperm(xb.size(0), device=xb.device)
                xb_mix = lam * xb + (1.0 - lam) * xb[perm]
                logits = model(xb_mix)
                loss = lam * loss_fn(logits, yb) + (1.0 - lam) * loss_fn(logits, yb[perm])
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_loss, val_pred, val_true = eval_pass(model, val_loader, loss_fn, device)
        val_f1 = f1_score(val_true, val_pred,
                          labels=list(range(num_classes)),
                          average="macro", zero_division=0)
        val_bal = balanced_accuracy_score(val_true, val_pred)
        tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_macro_f1": float(val_f1),
            "val_balanced_acc": float(val_bal),
        })
        log(f"  [seed={seed}] epoch {epoch:02d}  train_loss={tr_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  "
            f"val_bal={val_bal:.4f}  ({time.time() - t0:.1f}s)")

        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = float(val_f1)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log(f"  [seed={seed}] early stop at epoch {epoch} "
                    f"(best={best_epoch}, f1={best_val_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final test-set evaluation (CNN argmax) + 512-D feature extraction.
    test_loss, test_pred, test_true = eval_pass(model, test_loader, loss_fn, device)
    test_f1 = f1_score(test_true, test_pred,
                       labels=list(range(num_classes)),
                       average="macro", zero_division=0)
    test_acc = float((test_pred == test_true).mean())
    log(f"  [seed={seed}] FINAL test acc={test_acc:.4f}  macro-F1={test_f1:.4f}  "
        f"best_epoch={best_epoch}")

    return {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "val_macro_f1": float(best_val_f1),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "history": history,
        "state_dict": best_state,
        "split_idx": {
            "train": train_idx, "val": val_idx, "test": test_idx,
        },
    }


@torch.no_grad()
def eval_pass(model: nn.Module, loader: DataLoader,
              loss_fn: nn.Module, device: torch.device
              ) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(float(loss.detach().cpu()))
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_pred.append(pred)
        all_true.append(yb.detach().cpu().numpy())
    return (float(np.mean(losses)) if losses else float("nan"),
            np.concatenate(all_pred), np.concatenate(all_true))


@torch.no_grad()
def extract_backbone_features(model: nn.Module, specs: np.ndarray,
                              valid: np.ndarray, device: torch.device,
                              batch_size: int = 64) -> np.ndarray:
    """Run the fine-tuned backbone (everything except ``fc``) over all
    available samples and return a (N, 512) feature matrix."""
    model.eval()
    N = len(specs)
    feats = np.zeros((N, model.fc.in_features), dtype=np.float32)
    backbone = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4,
        model.avgpool, nn.Flatten(),
    ).to(device)
    for start in range(0, N, batch_size):
        chunk = specs[start:start + batch_size].astype(np.float32)
        t = torch.from_numpy(chunk).unsqueeze(1).to(device)
        feats[start:start + len(chunk)] = backbone(t).detach().cpu().numpy()
    feats[~valid] = 0.0
    return feats


# ---------------------------------------------------------------------------
# stage 3: shallow heads on fine-tuned features, + 4-class supplementary
# ---------------------------------------------------------------------------
def build_shallow_classifiers() -> dict[str, object]:
    clfs = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", n_jobs=-1, random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128), activation="relu",
            batch_size=128, max_iter=200, early_stopping=True,
            n_iter_no_change=15, random_state=42,
        ),
    }
    try:
        import xgboost as xgb
        clfs["XGBoost"] = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            objective="multi:softprob", tree_method="hist",
            n_jobs=-1, random_state=42, eval_metric="mlogloss",
        )
    except Exception as e:
        raise RuntimeError('XGBoost is required for this experiment.') from e
    return clfs


def eval_classifier(name: str, clf, X_test: np.ndarray, y_test: np.ndarray,
                    num_classes: int, class_keys: list[str]) -> tuple[dict, np.ndarray, np.ndarray | None]:
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        y_proba = None
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    rec = recall_score(y_test, y_pred, labels=list(range(num_classes)),
                       average=None, zero_division=0)
    prec = precision_score(y_test, y_pred, labels=list(range(num_classes)),
                           average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, labels=list(range(num_classes)),
                  average=None, zero_division=0)
    metrics = {
        "model": name,
        "overall_accuracy": float((y_pred == y_test).mean()),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }
    for c_idx, c_key in enumerate(class_keys):
        metrics[f"recall_{c_key}"] = float(rec[c_idx])
        metrics[f"precision_{c_key}"] = float(prec[c_idx])
        metrics[f"f1_{c_key}"] = float(f1[c_idx])
    return metrics, cm, y_proba


def write_classifier_suite(features: np.ndarray, valid: np.ndarray,
                           splits_df: pd.DataFrame,
                           cnn_metrics: dict,
                           cnn_cm: np.ndarray, cnn_proba: np.ndarray,
                           schema_tag: str, class_keys: list[str],
                           y_mapper: np.ndarray | None,
                           run_ensemble: bool = True,
                           cnn_validation_proba: np.ndarray | None = None) -> None:
    """Fit the four shallow heads on training features and evaluate validation/test files;
    serialize all artefacts.  ``schema_tag`` is "" for the 5-class main
    result (overwriting the Track-B filenames so figures keep working)
    and "_4class" for the supplementary run."""

    idx = splits_df.reset_index(drop=True)
    y_all = idx["class_idx"].to_numpy(dtype=np.int64)
    if y_mapper is not None:
        y_all = y_mapper[y_all]
    num_classes = len(class_keys)

    train_mask = ((idx["split"] == "train").to_numpy()) & valid
    val_mask = ((idx["split"] == "val").to_numpy()) & valid
    test_mask = ((idx["split"] == "test").to_numpy()) & valid

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_mask])
    X_validation = scaler.transform(features[val_mask])
    X_test = scaler.transform(features[test_mask])
    y_train = y_all[train_mask]
    y_validation = y_all[val_mask]
    y_test = y_all[test_mask]

    if cnn_validation_proba is None:
        raise ValueError('CNN validation predictions are required.')
    validation_metrics, validation_cm = metrics_from_proba('CNN', y_validation, cnn_validation_proba, class_keys)
    all_validation_metrics = [validation_metrics]
    validation_bank = {'CNN': cnn_validation_proba.astype(np.float64)}
    np.save(DATA_DIR / 'validation_labels.npy', y_validation)
    np.save(DATA_DIR / 'validation_proba_CNN.npy', cnn_validation_proba)
    np.save(DATA_DIR / 'validation_confusion_CNN.npy', validation_cm)
    for phase, mask in [('train', train_mask), ('validation', val_mask), ('test', test_mask)]:
        record_rows = idx.loc[mask, ['relative_path', 'class_key', 'start_time', 'end_time', 'sha256']].copy()
        record_rows.insert(0, 'record_index', np.flatnonzero(mask))
        record_rows['label_index'] = y_all[mask]
        record_rows.to_csv(DATA_DIR / f'{phase}_records.csv', index=False)

    all_metrics = [{**cnn_metrics, "model": "CNN"}]

    # persist CNN confusion / proba (for main schema we always keep both).
    np.save(DATA_DIR / f"confusion_CNN{schema_tag}.npy", cnn_cm)
    if cnn_proba is not None:
        np.save(DATA_DIR / f"proba_CNN{schema_tag}.npy", cnn_proba)
    if schema_tag == "":
        np.save(DATA_DIR / "test_labels.npy", y_test)

    # Collect per-model softmax proba for ensemble averaging.
    proba_bank: dict[str, np.ndarray] = {}
    if cnn_proba is not None:
        proba_bank["CNN"] = cnn_proba.astype(np.float64)

    clfs = build_shallow_classifiers()
    for name, clf in clfs.items():
        log(f"  [{schema_tag or 'main'}] fitting {name}...")
        t0 = time.time()
        try:
            clf.fit(X_train, y_train)
            joblib.dump({"classifier": clf, "scaler": scaler, "classes": class_keys,
                         "fit_subset": "train", "fit_indices": np.flatnonzero(train_mask)}, DATA_DIR / f"model_{name}{schema_tag}.joblib")
        except Exception as e:
            raise RuntimeError(f'Required model {name} failed.') from e
        dt = time.time() - t0
        m, cm, proba = eval_classifier(name, clf, X_test, y_test,
                                       num_classes, class_keys)
        m["fit_seconds"] = dt
        log(f"    {name}  acc={m['overall_accuracy']:.4f}  "
            f"macroF1={m['macro_f1']:.4f}  bal_acc={m['balanced_accuracy']:.4f}")
        np.save(DATA_DIR / f"confusion_{name}{schema_tag}.npy", cm)
        if proba is not None:
            np.save(DATA_DIR / f"proba_{name}{schema_tag}.npy", proba)
            proba_bank[name] = proba.astype(np.float64)
        all_metrics.append(m)
        mv, cmv, pv = eval_classifier(name, clf, X_validation, y_validation, num_classes, class_keys)
        if pv is None:
            raise RuntimeError(f'{name} did not return validation probabilities.')
        all_validation_metrics.append(mv)
        validation_bank[name] = pv.astype(np.float64)
        np.save(DATA_DIR / f'validation_proba_{name}.npy', pv)
        np.save(DATA_DIR / f'validation_confusion_{name}.npy', cmv)

    if not all_metrics:
        raise RuntimeError(f"{schema_tag}: no classifier succeeded")

    # ---- probability-averaging ensemble ----
    if run_ensemble and len(proba_bank) >= 2:
        members = globals().get("ENSEMBLE_MEMBERS", list(proba_bank))
        missing = set(members) - set(proba_bank)
        if missing:
            raise RuntimeError(f"Missing ensemble members: {missing}")
        stacked = np.stack([proba_bank[k] for k in members], axis=0)  # (K, N, C)
        mean_proba = stacked.mean(axis=0)
        y_pred_ens = mean_proba.argmax(axis=1)
        cm_ens = confusion_matrix(y_test, y_pred_ens,
                                  labels=list(range(num_classes)))
        rec_e = recall_score(y_test, y_pred_ens,
                             labels=list(range(num_classes)),
                             average=None, zero_division=0)
        prec_e = precision_score(y_test, y_pred_ens,
                                 labels=list(range(num_classes)),
                                 average=None, zero_division=0)
        f1_e = f1_score(y_test, y_pred_ens,
                        labels=list(range(num_classes)),
                        average=None, zero_division=0)
        m_ens = {
            "model": "Ensemble",
            "overall_accuracy": float((y_pred_ens == y_test).mean()),
            "macro_precision": float(prec_e.mean()),
            "macro_recall": float(rec_e.mean()),
            "macro_f1": float(f1_e.mean()),
            "balanced_accuracy":
                float(balanced_accuracy_score(y_test, y_pred_ens)),
            "fit_seconds": 0.0,
        }
        for c_idx, c_key in enumerate(class_keys):
            m_ens[f"recall_{c_key}"] = float(rec_e[c_idx])
            m_ens[f"precision_{c_key}"] = float(prec_e[c_idx])
            m_ens[f"f1_{c_key}"] = float(f1_e[c_idx])
        log(f"    Ensemble ({'+'.join(members)})  "
            f"acc={m_ens['overall_accuracy']:.4f}  "
            f"macroF1={m_ens['macro_f1']:.4f}  "
            f"bal_acc={m_ens['balanced_accuracy']:.4f}")
        np.save(DATA_DIR / f"confusion_Ensemble{schema_tag}.npy", cm_ens)
        np.save(DATA_DIR / f"proba_Ensemble{schema_tag}.npy",
                mean_proba.astype(np.float32))
        all_metrics.append(m_ens)
        pv = np.mean([validation_bank[k] for k in members], axis=0).astype(np.float32)
        mv, cmv = metrics_from_proba('Ensemble', y_validation, pv, class_keys)
        all_validation_metrics.append(mv)
        np.save(DATA_DIR / 'validation_proba_Ensemble.npy', pv)
        np.save(DATA_DIR / 'validation_confusion_Ensemble.npy', cmv)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = DATA_DIR / (f"multimodel_metrics{schema_tag}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(all_validation_metrics).to_csv(DATA_DIR / 'validation_metrics.csv', index=False)
    log(f"wrote {metrics_path}")

    # cross-model per-class F1 agreement (only for 5-class, consumed by the
    # existing plot_multimodel_consistency.py figure).
    if schema_tag == "":
        agree_rows = []
        for c_idx, c_key in enumerate(class_keys):
            vals = metrics_df[f"f1_{c_key}"].to_numpy()
            agree_rows.append({
                "class_key": c_key,
                "class_idx": c_idx,
                "f1_mean": float(vals.mean()),
                "f1_std": float(vals.std(ddof=0)),
                "f1_min": float(vals.min()),
                "f1_max": float(vals.max()),
            })
        pd.DataFrame(agree_rows).to_csv(
            DATA_DIR / "perclass_f1_agreement.csv", index=False,
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def metrics_from_proba(name, y, probabilities, class_keys):
    predictions = np.asarray(probabilities).argmax(axis=1)
    labels = list(range(len(class_keys)))
    cm = confusion_matrix(y, predictions, labels=labels)
    precision = precision_score(y, predictions, labels=labels, average=None, zero_division=0)
    recall = recall_score(y, predictions, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y, predictions, labels=labels, average=None, zero_division=0)
    row = {'model': name, 'overall_accuracy': float(np.mean(predictions == y)),
           'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
           'macro_f1': float(f1.mean()), 'balanced_accuracy': float(balanced_accuracy_score(y, predictions))}
    for i, key in enumerate(class_keys):
        row.update({f'precision_{key}':float(precision[i]), f'recall_{key}':float(recall[i]), f'f1_{key}':float(f1[i])})
    return row, cm


def predict_cnn_proba(model, specs, labels, indices, device, batch_size):
    ds = SpecDataset(specs, labels, indices, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval(); chunks = []
    with torch.no_grad():
        for xb, _ in loader:
            chunks.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(chunks)


def main():
    global DATA_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--trust-source", action="store_true", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True,
                        help="Verified within-class chronological split generated by split_manifest.py.")
    parser.add_argument("--ensemble-members", nargs="+", default=None)
    parser.add_argument("--data-root", required=True,
                        help="Path to the 5-class .npy archive. "
                             "The file inventory is checked against --split-manifest.")
    parser.add_argument("--export-heads-only", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild splits / spectrogram cache.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Default 32 on GPU / 64 on CPU.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEED_DEFAULTS))
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--skip-imagenet-baseline", action="store_true",
                        help="Skip Track-B feature export (speeds up debug).")
    parser.add_argument("--weight-mode", default="log",
                        choices=["log", "sqrt", "effective", "none"],
                        help="Loss class-weight mode; 'none' = plain CE.")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Focal modulator; 0 = plain CE.")
    parser.add_argument("--no-sampler", action="store_true",
                        help="Disable WeightedRandomSampler; use natural mini-batches.")
    parser.add_argument("--sampler-power", type=float, default=1.0 / 3.0,
                        help="WRS weight = 1/count^power; smaller = gentler.")
    parser.add_argument("--train-schema", default="5class",
                        choices=["5class", "4class"],
                        help="'4class' trains the CNN directly on the merged "
                             "natural-slope schema; 'multimodel_metrics.csv' "
                             "then reports the four-class results.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Beta(alpha, alpha) mixup strength on train "
                             "mini-batches. 0 disables, 0.2 is a good default.")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Skip the CNN + shallow-heads probability-average "
                             "ensemble row (on by default).")
    args = parser.parse_args()
    if args.epochs is None: args.epochs = 8 if args.train_schema == '5class' else 10
    if args.ensemble_members is None:
        args.ensemble_members = ['CNN', 'RandomForest', 'LogisticRegression', 'MLP', 'XGBoost'] if args.train_schema == '5class' else ['CNN', 'RandomForest']
    if args.out_dir.resolve().is_relative_to(Path(args.data_root).resolve()):
        parser.error('Output must be outside the source waveform directory.')
    DATA_DIR = args.out_dir.resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not args.data_root:
        parser.error("--data-root is required for provenance")
    root = Path(args.data_root).resolve()
    fixed_frame = load_manifest(root, args.split_manifest)
    files = sorted(root.rglob("*.npy"))
    fingerprint = hashlib.sha256()
    for p in files:
        fingerprint.update(str(p.relative_to(root)).replace("\\", "/").encode())
        fingerprint.update(hashlib.sha256(p.read_bytes()).digest())
    marker = DATA_DIR / "dataset_fingerprint.json"
    current = {"sha256": fingerprint.hexdigest(), "files": len(files), "data_root": str(root)}
    if marker.exists() and json.loads(marker.read_text())["sha256"] != current["sha256"]:
        raise RuntimeError("Result directory belongs to a different dataset; use a new --out-dir.")
    marker.write_text(json.dumps(current, indent=2), encoding="utf-8")
    versions = {name: importlib.metadata.version(name) for name in
                ['numpy', 'scipy', 'pandas', 'scikit-learn', 'xgboost', 'torch', 'torchvision', 'joblib']}
    versions['python'] = platform.python_version()
    config_for_hash = {k:v for k,v in vars(args).items() if k not in
                       ['out_dir', 'force', 'export_heads_only', 'trust_source', 'data_root', 'split_manifest']}
    signature = {'dataset_sha256': current['sha256'],
        'split_sha256': hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
        'settings': config_for_hash, 'versions': versions,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('split_manifest.py').read_bytes()).hexdigest(),
        'fit_subset': 'train', 'split_policy': fixed_frame.split_policy.iloc[0],
        'split_method': ('within_five_class_chronological_whole_day_next'
                         if fixed_frame.split_policy.iloc[0] == 'whole-day-next'
                         else 'within_five_class_chronological_7_2_1')}
    signature_path = DATA_DIR / 'run_signature.json'
    if signature_path.exists():
        assert_cache_signature(json.loads(signature_path.read_text('utf-8')), signature)
    elif (DATA_DIR/'splits.csv').exists():
        raise RuntimeError('Existing cache has no chronological run signature; use a new output directory.')
    signature_path.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    globals()["ENSEMBLE_MEMBERS"] = list(args.ensemble_members)
    (DATA_DIR / ("head_export_config.json" if args.export_heads_only else "run_config.json")).write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    if args.export_heads_only:
        frame=pd.read_csv(DATA_DIR / "splits.csv")
        valid=np.load(DATA_DIR / "valid_mask.npy")
        features=np.load(DATA_DIR / "features_finetuned.npy")
        mapper=MAP_5_TO_4 if args.train_schema=="4class" else None
        keys=CLASS_KEYS_4 if mapper is not None else CLASS_KEYS
        metrics=pd.read_csv(DATA_DIR / "multimodel_metrics.csv")
        cnn=metrics.loc[metrics["model"]=="CNN"].iloc[0].to_dict()
        write_classifier_suite(features,valid,frame,cnn,np.load(DATA_DIR / "confusion_CNN.npy"),np.load(DATA_DIR / "proba_CNN.npy"),"",keys,mapper,
                               cnn_validation_proba=np.load(DATA_DIR/'validation_proba_CNN.npy'))
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        setup_cpu_threads()
        log(f"device: CPU (torch threads={torch.get_num_threads()})")
    else:
        device = torch.device("cuda:0")
        log(f"device: CUDA ({torch.cuda.get_device_name(0)}, "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    if args.batch_size is None:
        args.batch_size = 32 if device.type == "cuda" else 64

    # ---- splits + spectrograms ----
    splits_path = DATA_DIR / "splits.csv"
    spec_path = DATA_DIR / "spectrograms.npy"
    if (splits_path.exists() and spec_path.exists() and not args.force):
        splits_df = pd.read_csv(splits_path)
        specs = np.load(spec_path, mmap_mode="r")
        valid = np.load(DATA_DIR / "valid_mask.npy")
        log(f"cache hit: splits ({len(splits_df)}) + specs ({specs.shape}) "
            f"+ valid ({int(valid.sum())})")
    else:
        if not args.data_root:
            raise SystemExit(
                "spectrogram cache missing; pass --data-root to rebuild."
            )
        splits_df, specs, valid = ensure_cache(Path(args.data_root), args.force, fixed_frame)

    columns = ['relative_path', 'class_key', 'class_idx', 'start_time', 'end_time', 'split', 'sha256', 'split_policy']
    if not splits_df[columns].equals(fixed_frame[columns]) or len(specs) != len(fixed_frame) or not valid.all():
        raise RuntimeError('Cached rows or validity flags do not match the fixed manifest.')
    if not np.isfinite(specs).all():
        raise ValueError('Non-finite spectrogram input.')
    (DATA_DIR/'environment.json').write_text(json.dumps({'versions':versions, 'device':str(device),
      'device_name':torch.cuda.get_device_name(0) if device.type=='cuda' else platform.processor(),
      'cuda':torch.version.cuda,'deterministic_algorithms':True},indent=2),encoding='utf-8')

    # ---- Track-B baseline features (for the ImageNet-vs-finetuned ablation) ----
    if not args.skip_imagenet_baseline:
        feat_imagenet = export_imagenet_features(specs, valid, device, args.force)
    else:
        feat_imagenet = None

    # ---- resolve training schema ----
    if args.train_schema == "4class":
        active_keys = CLASS_KEYS_4
        active_mapper = MAP_5_TO_4
        log(f"TRAIN SCHEMA = 4-class (natural_slope_events = microseismic "
            f"+ small_landslide); CNN outputs {len(active_keys)} logits.")
    else:
        active_keys = CLASS_KEYS
        active_mapper = None
        log(f"TRAIN SCHEMA = 5-class (baseline); CNN outputs "
            f"{len(active_keys)} logits.")

    # ---- multi-seed fine-tune ----
    all_runs = []
    for seed in args.seeds:
        log(f"=== fine-tuning (seed={seed}) ===")
        run = train_one_seed(
            specs, valid, splits_df, device, seed,
            epochs=args.epochs, batch_size=args.batch_size,
            lr_head=args.lr_head, lr_backbone=args.lr_backbone,
            weight_decay=args.weight_decay, patience=args.patience,
            weight_mode=args.weight_mode, focal_gamma=args.focal_gamma,
            use_sampler=(not args.no_sampler),
            sampler_power=args.sampler_power,
            class_keys=active_keys, label_mapper=active_mapper,
            mixup_alpha=args.mixup_alpha,
        )
        all_runs.append(run)

    # pick median seed by val macro-F1
    all_runs.sort(key=lambda r: r["val_macro_f1"])
    median_run = all_runs[len(all_runs) // 2]
    log(f"median seed = {median_run['seed']}  "
        f"val_f1={median_run['val_macro_f1']:.4f}")

    # export training histories
    hist_median_df = pd.DataFrame(median_run["history"])
    hist_median_df.to_csv(DATA_DIR / "training_history.csv", index=False)
    (DATA_DIR / "finetune_history.json").write_text(
        json.dumps(
            [{"seed": r["seed"], "history": r["history"],
              "val_macro_f1": r["val_macro_f1"],
              "test_acc": r["test_acc"],
              "test_macro_f1": r["test_macro_f1"],
              "best_epoch": r["best_epoch"]}
             for r in all_runs],
            indent=2,
        ),
        encoding="utf-8",
    )

    # reload the median-seed best model and regenerate the authoritative
    # test-set predictions + backbone features
    model = build_resnet18_1ch(len(active_keys), imagenet=False).to(device)
    model.load_state_dict(median_run["state_dict"])
    torch.save({"state_dict": median_run["state_dict"], "class_keys": active_keys, "seed": median_run["seed"], "best_epoch": median_run["best_epoch"], "dataset": current,
                "split_sha256":signature['split_sha256'],"fit_subset":"train", "fit_indices":median_run['split_idx']['train'].tolist()}, DATA_DIR / "cnn_checkpoint.pt")

    y_raw_full = splits_df["class_idx"].to_numpy(dtype=np.int64)
    y_active_full = (active_mapper[y_raw_full] if active_mapper is not None
                     else y_raw_full)

    loss_fn = ClassBalancedFocalLoss(
        np.bincount(y_active_full[median_run['split_idx']['train']], minlength=len(active_keys)).astype(np.float32)
    ).to(device)
    test_idx = median_run["split_idx"]["test"]
    test_ds = SpecDataset(specs, y_active_full, test_idx, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0,
                             pin_memory=(device.type == "cuda"))
    _, cnn_pred, cnn_true = eval_pass(model, test_loader, loss_fn, device)
    with torch.no_grad():
        all_logits = []
        model.eval()
        for xb, _ in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            all_logits.append(logits.detach().cpu().numpy())
    cnn_logits = np.concatenate(all_logits)
    cnn_proba = np.exp(cnn_logits - cnn_logits.max(axis=1, keepdims=True))
    cnn_proba /= cnn_proba.sum(axis=1, keepdims=True)

    # CNN metrics on the active schema
    cm_cnn = confusion_matrix(cnn_true, cnn_pred,
                              labels=list(range(len(active_keys))))
    rec = recall_score(cnn_true, cnn_pred,
                       labels=list(range(len(active_keys))),
                       average=None, zero_division=0)
    prec = precision_score(cnn_true, cnn_pred,
                           labels=list(range(len(active_keys))),
                           average=None, zero_division=0)
    f1 = f1_score(cnn_true, cnn_pred,
                  labels=list(range(len(active_keys))),
                  average=None, zero_division=0)
    cnn_metrics = {
        "overall_accuracy": float((cnn_pred == cnn_true).mean()),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(cnn_true, cnn_pred)),
        "fit_seconds": 0.0,
    }
    for c_idx, c_key in enumerate(active_keys):
        cnn_metrics[f"recall_{c_key}"] = float(rec[c_idx])
        cnn_metrics[f"precision_{c_key}"] = float(prec[c_idx])
        cnn_metrics[f"f1_{c_key}"] = float(f1[c_idx])
    log(f"CNN ({args.train_schema})  test acc={cnn_metrics['overall_accuracy']:.4f}  "
        f"macro-F1={cnn_metrics['macro_f1']:.4f}  "
        f"bal_acc={cnn_metrics['balanced_accuracy']:.4f}")

    # ---- shallow-head suite on fine-tuned features ----
    log("extracting fine-tuned 512-D backbone features on all samples...")
    feat_finetuned = extract_backbone_features(model, specs, valid, device,
                                               batch_size=args.batch_size)
    np.save(DATA_DIR / "features_finetuned.npy", feat_finetuned)
    # keep compatibility: some existing scripts look for "features.npy".
    np.save(DATA_DIR / "features.npy", feat_finetuned)
    log("wrote features_finetuned.npy (also duplicated as features.npy)")

    log(f"=== {args.train_schema} cross-model consistency on fine-tuned features ===")
    cnn_val_proba = predict_cnn_proba(model, specs, y_active_full, median_run['split_idx']['val'], device, args.batch_size)
    write_classifier_suite(feat_finetuned, valid, splits_df,
                           cnn_metrics, cm_cnn, cnn_proba,
                           schema_tag="", class_keys=active_keys,
                           y_mapper=active_mapper,
                           run_ensemble=(not args.no_ensemble), cnn_validation_proba=cnn_val_proba)

    # The other schema is trained independently in its own output directory.

    # ---- ImageNet-vs-finetuned comparison (headline ablation) ----
    if feat_imagenet is not None:
        log("=== ImageNet-frozen vs fine-tuned ablation ===")
        idx = splits_df.reset_index(drop=True)
        y_all_abl = (active_mapper[idx["class_idx"].to_numpy(dtype=np.int64)]
                     if active_mapper is not None
                     else idx["class_idx"].to_numpy(dtype=np.int64))
        train_mask = ((idx["split"] == "train").to_numpy()) & valid
        val_mask = ((idx["split"] == "val").to_numpy()) & valid
        test_mask = ((idx["split"] == "test").to_numpy()) & valid
        y_train = y_all_abl[train_mask]
        y_test = y_all_abl[test_mask]

        rows = []
        for tag, feats in (("imagenet_frozen", feat_imagenet),
                           ("finetuned", feat_finetuned)):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(feats[train_mask])
            Xte = scaler.transform(feats[test_mask])
            rf = RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1, random_state=42,
            )
            rf.fit(Xtr, y_train)
            joblib.dump({'classifier':rf,'scaler':scaler,'classes':active_keys,'fit_subset':'train',
                         'fit_indices':np.flatnonzero(train_mask)},DATA_DIR/f'ablation_{tag}.joblib')
            y_pred = rf.predict(Xte)
            for phase, mask in [('test',test_mask),('validation',val_mask)]:
                probability=rf.predict_proba(scaler.transform(feats[mask]))
                np.save(DATA_DIR/f'ablation_{tag}_{phase}_proba.npy',probability)
                scored,_=metrics_from_proba('RandomForest',y_all_abl[mask],probability,active_keys)
                (DATA_DIR/f'ablation_{tag}_{phase}_metrics.json').write_text(json.dumps(scored,indent=2),encoding='utf-8')
            rows.append({
                "representation": tag,
                "model": "RandomForest",
                "overall_accuracy": float((y_pred == y_test).mean()),
                "macro_f1": float(f1_score(y_test, y_pred,
                                           labels=list(range(len(active_keys))),
                                           average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            })
        rows.append({
            "representation": "finetuned",
            "model": "CNN_argmax",
            "overall_accuracy": cnn_metrics["overall_accuracy"],
            "macro_f1": cnn_metrics["macro_f1"],
            "balanced_accuracy": cnn_metrics["balanced_accuracy"],
        })
        pd.DataFrame(rows).to_csv(
            DATA_DIR / "imagenet_vs_finetuned_comparison.csv", index=False,
        )

    # ---- summary JSON ----
    metrics5 = pd.read_csv(DATA_DIR / "multimodel_metrics.csv")
    summary = {
        "n_samples_total": int(len(splits_df)),
        "n_valid": int(valid.sum()),
        "n_train": int(((splits_df["split"] == "train") & valid).sum()),
        "n_val": int(((splits_df["split"] == "val") & valid).sum()),
        "n_test": int(((splits_df["split"] == "test") & valid).sum()),
        "train_schema": args.train_schema,
        "split_method": signature['split_method'],
        "split_policy": signature['split_policy'],
        "split_sha256": signature['split_sha256'],
        "fit_subset": "train",
        "ensemble_members": args.ensemble_members,
        "best_epoch": int(median_run['best_epoch']),
        "classes": active_keys,
        "class_display_names": (
            CLASS_DIRS if args.train_schema == "5class"
            else ["natural slope events", "blasting",
                  "mechanical_mining", "transport_vehicles"]
        ),
        "per_class_total": [
            int((y_active_full == c).sum()) for c in range(len(active_keys))
        ],
        "per_class_test": [
            int((cnn_true == c).sum()) for c in range(len(active_keys))
        ],
        "models": metrics5["model"].tolist(),
        "mean_overall_accuracy": float(metrics5["overall_accuracy"].mean()),
        "std_overall_accuracy": float(metrics5["overall_accuracy"].std(ddof=0)),
        "mean_macro_f1": float(metrics5["macro_f1"].mean()),
        "std_macro_f1": float(metrics5["macro_f1"].std(ddof=0)),
        "mean_balanced_accuracy": float(metrics5["balanced_accuracy"].mean()),
        "std_balanced_accuracy": float(
            metrics5["balanced_accuracy"].std(ddof=0)
        ),
        "seeds": list(args.seeds),
        "median_seed": int(median_run["seed"]),
        "cnn_test_acc_per_seed": {
            int(r["seed"]): float(r["test_acc"]) for r in all_runs
        },
        "cnn_test_f1_per_seed": {
            int(r["seed"]): float(r["test_macro_f1"]) for r in all_runs
        },
    }
    (DATA_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    log(f"wrote {DATA_DIR / 'summary.json'}")
    log("all stages complete.")


if __name__ == "__main__":
    sys.exit(main())
