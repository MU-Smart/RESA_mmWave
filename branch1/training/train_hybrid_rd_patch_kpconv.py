"""Train a KPConv temporal segmenter with hybrid RD/RA patch augmentation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

# import path settings from config.config (includes load_mod helper)
from config.config import REPO_ROOT, load_mod


hybrid_rd_model = load_mod("hybrid_rd_model", "branch1/models/hybrid_rd/hybrid_rd_model.py")
RDPatchTemporalSegmenter = hybrid_rd_model.RDPatchTemporalSegmenter

hybrid_rd_ds = load_mod("hybrid_rd_temporal_dataset", "branch1/models/hybrid_rd/hybrid_rd_temporal_dataset.py")
HybridRDTemporalEvalDataset = hybrid_rd_ds.HybridRDTemporalEvalDataset
HybridRDTemporalTrainDataset = hybrid_rd_ds.HybridRDTemporalTrainDataset
hybrid_rd_eval_collate_fn = hybrid_rd_ds.hybrid_rd_eval_collate_fn

kpconv_mod = load_mod("radar_3dgcnn_common_kpconv", "branch1/models/3dgcnn/radar_3dgcnn_common_kpconv.py")
KPConvTemporalSegmenter = kpconv_mod.KPConvTemporalSegmenter

BASE_FEATURE_COLS = [
    "x", "y", "z",
    "range_m", "azimuth_deg", "elevation_deg",
    "doppler", "snr",
    "local_density", "persist_score",
    "z_norm_range", "frame_doppler_abs", "frame_doppler_std",
]
SOFT_STRUCTURAL_COLS = [
    "z_above_floor", "corridor_margin", "wall_anomaly",
    "floor_ang", "wall_ang", "has_floor", "has_wall",
]
DIRECTNESS_COLS = [
    "ego_doppler_residual_mps", "ego_residual_abs_z",
    "ego_inlier_flag", "p_ego", "p_dir_doppler",
]
RD_SCALAR_COLS = [
    "rd_entropy", "rd_doppler_spread", "rd_anisotropy", "rd_peak_ratio",
]
FEATURE_MODES = {
    "base": BASE_FEATURE_COLS,
    "soft_structural": BASE_FEATURE_COLS + SOFT_STRUCTURAL_COLS,
    "directness_soft_structural": BASE_FEATURE_COLS + SOFT_STRUCTURAL_COLS + DIRECTNESS_COLS + RD_SCALAR_COLS,
}
RAW_TO_3CLASS = {
    "wall": "structure",
    "door": "structure",
    "pillar": "structure",
    "box_like": "structure",
    "floor": "floor",
    "human": "human",
    "structure": "structure",
}


def add_derived_directness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ego-directness derived columns from ego_doppler_residual_mps when available.

    Derives ego_residual_abs_z, ego_inlier_flag, p_ego, and p_dir_doppler using
    frame-level residual MAD normalization.  Safe no-op if the source column is absent.
    """
    if "ego_doppler_residual_mps" not in df.columns:
        return df
    # Already fully derived — nothing to do
    if all(c in df.columns for c in ["ego_residual_abs_z", "p_dir_doppler"]):
        return df
    df = df.copy()
    res = df["ego_doppler_residual_mps"].astype(np.float32).values
    # Frame-level sigma: 1.4826 * MAD of static (structure/floor) point residuals
    group_cols = [c for c in ["session", "radar_frame_num"] if c in df.columns]
    if group_cols and "bucket_3class" in df.columns:
        sigma = np.full(len(df), 0.35, dtype=np.float32)
        static_mask = df["bucket_3class"].isin(["structure", "floor"]).values
        groups = df.groupby(group_cols, sort=False)
        for key, idx in groups.groups.items():
            s_idx = idx[static_mask[idx]]
            if len(s_idx) >= 3:
                mad = float(np.median(np.abs(res[s_idx] - np.median(res[s_idx]))))
                sigma[idx] = max(1.4826 * mad, 0.05)
    else:
        global_mad = float(np.median(np.abs(res - np.median(res))))
        sigma = np.full(len(df), max(1.4826 * global_mad, 0.05), dtype=np.float32)
    abs_z = np.abs(res) / sigma
    abs_z_clip = np.clip(abs_z, 0.0, 4.0)
    if "ego_residual_abs_z" not in df.columns:
        df["ego_residual_abs_z"] = abs_z.astype(np.float32)
    if "ego_inlier_flag" not in df.columns:
        df["ego_inlier_flag"] = (abs_z < 2.0).astype(np.float32)
    if "p_ego" not in df.columns:
        if "ego_available" in df.columns:
            df["p_ego"] = df["ego_available"].astype(float).fillna(0.0).astype(np.float32)
        else:
            df["p_ego"] = np.float32(1.0)
    if "p_dir_doppler" not in df.columns:
        df["p_dir_doppler"] = np.exp(-0.5 * abs_z_clip ** 2).astype(np.float32)
    return df


def resolve_feature_cols(args: argparse.Namespace) -> list[str]:
    if args.feature_cols:
        return [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    return list(FEATURE_MODES[args.feature_mode])


def parse_bucket_order(raw: str) -> list[str]:
    out = [x.strip() for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("--bucket-order must not be empty.")
    return out


def ensure_label_column(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    out = df.copy()
    if label_col in out.columns:
        if label_col == "bucket_3class" and "bucket" in out.columns:
            repaired = out["bucket"].map(RAW_TO_3CLASS)
            missing = out[label_col].isna() | ~out[label_col].isin(RAW_TO_3CLASS.values())
            out.loc[missing, label_col] = repaired[missing]
        return out
    if label_col == "bucket_3class" and "bucket" in out.columns:
        out[label_col] = out["bucket"].map(RAW_TO_3CLASS)
        return out
    raise ValueError(f"Dataset missing label column '{label_col}'.")


def load_dataset(path: str | Path, *, feature_cols: list[str], label_col: str, bucket_order: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = ensure_label_column(df, label_col)
    if "rd_patch_valid" in df.columns:
        df = df[df["rd_patch_valid"].astype(bool)].copy()
    if "ra_patch_valid" in df.columns:
        df = df[df["ra_patch_valid"].astype(bool)].copy()
    df = add_derived_directness_features(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    df = df[df[label_col].isin(bucket_order)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No usable rows after label/patch filtering: {path}")
    return df


def validate_split_contract(df: pd.DataFrame, *, require_val: bool) -> None:
    if "split" not in df.columns:
        raise ValueError("--dataset must contain split column with train/val.")
    splits = set(str(x) for x in df["split"].dropna().unique().tolist())
    missing = {"train"} - splits
    if require_val:
        missing |= {"val"} - splits
    if missing:
        raise ValueError(f"Dataset split column is missing required split(s): {sorted(missing)}")


def compute_feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = df[feature_cols].mean().to_numpy(dtype=np.float32)
    stds = df[feature_cols].std().to_numpy(dtype=np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)
    return means, stds


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, bucket_order: list[str]) -> dict[str, object]:
    preds = logits.argmax(dim=1)
    targets_np = targets.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()
    acc = float((preds == targets).float().mean().item())
    bal = float(balanced_accuracy_score(targets_np, preds_np)) if len(np.unique(targets_np)) > 1 else acc
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np,
        preds_np,
        labels=list(range(len(bucket_order))),
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "per_class_precision": {b: float(precision[i]) for i, b in enumerate(bucket_order)},
        "per_class_recall": {b: float(recall[i]) for i, b in enumerate(bucket_order)},
        "per_class_f1": {b: float(f1[i]) for i, b in enumerate(bucket_order)},
    }


class FocalLoss(nn.Module):
    """Per-point focal loss with optional class weights and reduction='none'."""

    def __init__(self, *, weight: torch.Tensor | None = None, gamma: float = 1.5) -> None:
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone().float())
        else:
            self.weight = None
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        if self.gamma <= 0.0:
            return ce
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.reshape(-1, 1)).squeeze(1)
        pt = log_pt.exp().clamp(min=1e-6, max=1.0)
        return ((1.0 - pt) ** self.gamma) * ce


def build_balanced_window_sampler(
    dataset: HybridRDTemporalTrainDataset,
    *,
    class_counts: np.ndarray,
    power: float,
    max_weight: float,
) -> tuple[WeightedRandomSampler, dict[str, float]]:
    counts = np.maximum(np.asarray(class_counts, dtype=np.float64), 1.0)
    ratios = (counts.max() / counts) ** float(power)
    ratios = np.clip(ratios, 1.0, float(max_weight))
    weights = np.ones(len(dataset.windows), dtype=np.float64)
    for i, win in enumerate(dataset.windows):
        center = dataset.frames[win.center_index]
        labels = dataset.y[center.point_indices]
        if labels.size:
            weights[i] = float(np.max(ratios[np.unique(labels)]))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    stats = {
        "min": float(weights.min()) if weights.size else 0.0,
        "max": float(weights.max()) if weights.size else 0.0,
        "mean": float(weights.mean()) if weights.size else 0.0,
    }
    return sampler, stats


def flat_logits_targets(logits: torch.Tensor, labels: torch.Tensor, n_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    return logits.transpose(1, 2).reshape(-1, n_classes), labels.reshape(-1)


def split_model_output(output: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, dict):
        return output["sem_logits"], output.get("acc_logits")
    return output, None


def has_ra_patch_columns(df: pd.DataFrame) -> bool:
    return {"ra_patch_valid", "ra_patch_shard", "ra_patch_index"}.issubset(df.columns)


def load_patch_manifest(dataset_root: str | Path) -> dict[str, object]:
    manifest_path = Path(dataset_root) / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def compute_acc_loss(
    acc_logits: torch.Tensor | None,
    acc_targets: torch.Tensor | None,
    acc_mask: torch.Tensor | None,
    bce_criterion: nn.Module,
    point_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, int]:
    if acc_logits is None or acc_targets is None or acc_mask is None:
        return None, 0
    logits = acc_logits.squeeze(1)
    targets = acc_targets.float()
    mask = acc_mask.float()
    if point_weights is not None:
        mask = mask * point_weights.reshape_as(mask).float()
    denom = mask.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return logits.sum() * 0.0, 0
    loss = bce_criterion(logits, targets)
    return (loss * mask).sum() / denom.clamp(min=1e-6), int((acc_mask > 0).sum().item())


def compute_acc_metrics(
    logits: torch.Tensor | None,
    targets: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> dict[str, object]:
    if logits is None or targets is None or mask is None:
        return {}
    flat_mask = mask.reshape(-1) > 0
    n = int(flat_mask.sum().item())
    if n == 0:
        return {"acc_n": 0}
    logits_flat = logits.squeeze(1).reshape(-1) if logits.ndim >= 3 else logits.reshape(-1)
    scores_t = torch.sigmoid(logits_flat[flat_mask]).detach().cpu()
    targets_t = targets.reshape(-1)[flat_mask].detach().cpu().float()
    scores = scores_t.numpy()
    target_np = targets_t.numpy().astype(np.int64)
    preds = (scores >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_np,
        preds,
        labels=[1],
        zero_division=0,
    )
    cm = confusion_matrix(target_np, preds, labels=[0, 1])
    try:
        auc = float(roc_auc_score(target_np, scores)) if len(np.unique(target_np)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "acc_n": n,
        "acc_accuracy": float((preds == target_np).mean()),
        "acc_auc": auc,
        "acc_precision": float(precision[0]),
        "acc_recall": float(recall[0]),
        "acc_f1": float(f1[0]),
        "acc_confusion": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def run_train_epoch(
    model,
    loader,
    criterion,
    acc_criterion,
    optimizer,
    device,
    n_classes: int,
    bucket_order: list[str],
    *,
    use_ra_patches: bool = False,
    acc_loss_weight: float = 0.0,
) -> dict[str, object]:
    model.train(True)
    total_loss = 0.0
    total_sem_loss = 0.0
    total_acc_loss = 0.0
    total_acc_weight = 0
    all_logits = []
    all_targets = []
    all_acc_logits = []
    all_acc_targets = []
    all_acc_masks = []
    for batch in loader:
        if use_ra_patches:
            window_features, window_xyz, center_labels, frame_meta, rd_patches, ra_patches, sample_weights = batch[:7]
            offset = 7
        else:
            window_features, window_xyz, center_labels, frame_meta, rd_patches, sample_weights = batch[:6]
            ra_patches = None
            offset = 6
        acc_targets = batch[offset] if len(batch) > offset else None
        acc_mask = batch[offset + 1] if len(batch) > offset + 1 else None
        window_features = window_features.to(device)
        window_xyz = window_xyz.to(device)
        center_labels = center_labels.to(device)
        frame_meta = frame_meta.to(device)
        rd_patches = rd_patches.to(device)
        if ra_patches is not None:
            ra_patches = ra_patches.to(device)
        sample_weights = sample_weights.to(device)
        flat_weights = sample_weights.reshape(-1)
        if acc_targets is not None:
            acc_targets = acc_targets.to(device)
        if acc_mask is not None:
            acc_mask = acc_mask.to(device)

        model_out = model(window_features, window_xyz, frame_meta, rd_patches, ra_patches)
        logits, acc_logits = split_model_output(model_out)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if acc_logits is not None:
            acc_logits = torch.nan_to_num(acc_logits, nan=0.0, posinf=0.0, neginf=0.0)
        loss_logits, loss_targets = flat_logits_targets(logits, center_labels, n_classes)
        per_point_loss = criterion(loss_logits, loss_targets)
        sem_loss = (per_point_loss * flat_weights).sum() / flat_weights.sum().clamp(min=1e-6)
        acc_loss, acc_n = compute_acc_loss(acc_logits, acc_targets, acc_mask, acc_criterion, sample_weights)
        loss = sem_loss if acc_loss is None else sem_loss + float(acc_loss_weight) * acc_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * int(loss_targets.numel())
        total_sem_loss += float(sem_loss.item()) * int(loss_targets.numel())
        if acc_loss is not None:
            total_acc_loss += float(acc_loss.item()) * max(acc_n, 1)
            total_acc_weight += acc_n
        all_logits.append(loss_logits.detach())
        all_targets.append(loss_targets.detach())
        if acc_logits is not None and acc_targets is not None and acc_mask is not None:
            all_acc_logits.append(acc_logits.detach().squeeze(1).reshape(-1))
            all_acc_targets.append(acc_targets.detach().reshape(-1))
            all_acc_masks.append(acc_mask.detach().reshape(-1))

    logits_t = torch.cat(all_logits, dim=0)
    targets_t = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_t, targets_t, bucket_order)
    metrics["loss"] = float(total_loss / max(int(targets_t.numel()), 1))
    metrics["sem_loss"] = float(total_sem_loss / max(int(targets_t.numel()), 1))
    if all_acc_logits:
        metrics["acc_loss"] = float(total_acc_loss / max(total_acc_weight, 1))
        metrics.update(compute_acc_metrics(
            torch.cat(all_acc_logits, dim=0),
            torch.cat(all_acc_targets, dim=0),
            torch.cat(all_acc_masks, dim=0),
        ))
    return metrics


@torch.no_grad()
def run_eval_epoch(
    model,
    loader,
    criterion,
    acc_criterion,
    device,
    n_classes: int,
    bucket_order: list[str],
    *,
    use_ra_patches: bool = False,
    acc_loss_weight: float = 0.0,
) -> dict[str, object]:
    model.train(False)
    total_loss = 0.0
    total_sem_loss = 0.0
    total_acc_loss = 0.0
    total_acc_weight = 0
    all_logits = []
    all_targets = []
    all_acc_logits = []
    all_acc_targets = []
    all_acc_masks = []
    for batch in loader:
        feat_list = [t.unsqueeze(0).to(device) for t in batch["window_features"]]
        xyz_list = [t.unsqueeze(0).to(device) for t in batch["window_xyz"]]
        patch_list = [t.unsqueeze(0).to(device) for t in batch["window_rd_patches"]]
        ra_patch_list = None
        if use_ra_patches:
            if "window_ra_patches" not in batch:
                raise ValueError("RA-aware checkpoint requires window_ra_patches in the eval dataset.")
            ra_patch_list = [t.unsqueeze(0).to(device) for t in batch["window_ra_patches"]]
        frame_meta = batch["frame_meta"].unsqueeze(0).to(device)
        labels = batch["center_labels"].unsqueeze(0).to(device)
        acc_targets = batch.get("acc_targets")
        acc_mask = batch.get("acc_mask")
        if acc_targets is not None:
            acc_targets = acc_targets.unsqueeze(0).to(device)
        if acc_mask is not None:
            acc_mask = acc_mask.unsqueeze(0).to(device)
        model_out = model(feat_list, xyz_list, frame_meta, patch_list, ra_patch_list)
        logits, acc_logits = split_model_output(model_out)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if acc_logits is not None:
            acc_logits = torch.nan_to_num(acc_logits, nan=0.0, posinf=0.0, neginf=0.0)
        loss_logits, loss_targets = flat_logits_targets(logits, labels, n_classes)
        sem_loss = criterion(loss_logits, loss_targets)
        acc_loss, acc_n = compute_acc_loss(acc_logits, acc_targets, acc_mask, acc_criterion)
        total_sem_loss += float(sem_loss.sum().item())
        total_loss += float(sem_loss.sum().item())
        if acc_loss is not None:
            total_loss += float(acc_loss_weight) * float(acc_loss.item()) * max(acc_n, 1)
            total_acc_loss += float(acc_loss.item()) * max(acc_n, 1)
            total_acc_weight += acc_n
        all_logits.append(loss_logits.cpu())
        all_targets.append(loss_targets.cpu())
        if acc_logits is not None and acc_targets is not None and acc_mask is not None:
            all_acc_logits.append(acc_logits.cpu().squeeze(1).reshape(-1))
            all_acc_targets.append(acc_targets.cpu().reshape(-1))
            all_acc_masks.append(acc_mask.cpu().reshape(-1))

    logits_t = torch.cat(all_logits, dim=0)
    targets_t = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_t, targets_t, bucket_order)
    metrics["loss"] = float(total_loss / max(int(targets_t.numel()), 1))
    metrics["sem_loss"] = float(total_sem_loss / max(int(targets_t.numel()), 1))
    if all_acc_logits:
        metrics["acc_loss"] = float(total_acc_loss / max(total_acc_weight, 1))
        metrics.update(compute_acc_metrics(
            torch.cat(all_acc_logits, dim=0),
            torch.cat(all_acc_targets, dim=0),
            torch.cat(all_acc_masks, dim=0),
        ))
    return metrics


def metric_value(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name, float("nan"))
    return float(value) if isinstance(value, (int, float, np.floating)) else float("nan")


def better(best_metric: str, current: float, best: float) -> bool:
    if np.isnan(current):
        return False
    if np.isnan(best):
        return True
    return current < best if best_metric.endswith("loss") else current > best


def main() -> None:
    ap = argparse.ArgumentParser(description="Train hybrid RD/RA-patch KPConv point-cloud model.")
    ap.add_argument("--dataset", required=True, type=Path, help="points_with_rd_patch_index.csv for train/internal val.")
    ap.add_argument("--dataset-root", required=True, type=Path, help="Directory containing patch shards.")
    ap.add_argument("--external-val-dataset", type=Path, default=None)
    ap.add_argument("--external-val-root", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--resume-checkpoint", type=Path, default=None,
                    help="Resume model, optimizer, scheduler, and best metric state from a previous checkpoint.")
    ap.add_argument("--feature-stats-checkpoint", type=Path, default=None,
                    help="Reuse feature normalization stats from a checkpoint. Defaults to --resume-checkpoint when resuming.")
    ap.add_argument("--label-col", default="bucket_3class")
    ap.add_argument("--bucket-order", default="structure,floor,human")
    ap.add_argument("--feature-mode", default="soft_structural", choices=sorted(FEATURE_MODES))
    ap.add_argument("--feature-cols", default=None, help="Comma-separated explicit feature columns.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--windows-per-batch", type=int, default=4)
    ap.add_argument("--points-per-frame", type=int, default=64)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--emb-dims", type=int, default=256)
    ap.add_argument("--temporal-layers", type=int, default=2)
    ap.add_argument("--temporal-heads", type=int, default=4)
    ap.add_argument("--gate-hidden", type=int, default=128)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--radius-1", type=float, default=0.35)
    ap.add_argument("--radius-2", type=float, default=0.50)
    ap.add_argument("--radius-3", type=float, default=0.70)
    ap.add_argument("--sigma-factor", type=float, default=2.5)
    ap.add_argument("--n-kernel-points", type=int, default=15)
    ap.add_argument("--rd-patch-embed-dim", type=int, default=32)
    ap.add_argument("--rd-patch-channels", type=int, default=1)
    ap.add_argument("--rd-patch-doppler-bins", type=int, default=17)
    ap.add_argument("--rd-patch-range-bins", type=int, default=7)
    ap.add_argument("--patch-fusion-mode", default="cnn", choices=["cnn", "attention"],
                    help="How each point's RD/RA patch becomes an embedding. 'cnn' (default) matches every "
                         "existing checkpoint. 'attention' tokenizes patch cells and runs a small "
                         "Transformer over them instead of a CNN + average-pool.")
    ap.add_argument("--best-metric", default="val_bal_acc", choices=["val_loss", "val_bal_acc", "external_val_loss", "external_val_bal_acc"])
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--loss-kind", default="cross_entropy", choices=["cross_entropy", "focal"],
                    help="Semantic loss for the point classifier. Focal keeps the same class weights but focuses hard examples.")
    ap.add_argument("--focal-gamma", type=float, default=1.5,
                    help="Gamma for --loss-kind focal. Ignored for cross_entropy.")
    ap.add_argument("--balanced-window-sampler", action="store_true",
                    help="Oversample temporal windows whose center frame contains rare classes.")
    ap.add_argument("--window-sampler-power", type=float, default=0.5,
                    help="Power applied to inverse class-frequency ratios for --balanced-window-sampler.")
    ap.add_argument("--window-sampler-max-weight", type=float, default=20.0,
                    help="Maximum per-window sampling weight for --balanced-window-sampler.")
    ap.add_argument("--sample-weight-col", default=None, help="Column in dataset CSV for per-sample loss weights (e.g. final_train_weight). Weights are clipped to [0, 1]. Omit for uniform weighting.")
    ap.add_argument("--has-acc-head", action="store_true", help="Train the optional accumulatability head.")
    ap.add_argument("--acc-label-col", default="accumulatable_label", help="Column containing yes/no/uncertain accumulatability labels.")
    ap.add_argument("--acc-loss-weight", type=float, default=0.25, help="Weight for masked accumulatability BCE loss when --has-acc-head is enabled.")
    ap.add_argument("--acc-head-hidden-dim", type=int, default=64, help="Hidden width for the accumulatability MLP head.")
    args = ap.parse_args()

    if args.external_val_dataset and not args.external_val_root:
        raise ValueError("--external-val-root is required with --external-val-dataset.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    bucket_order = parse_bucket_order(args.bucket_order)
    n_classes = len(bucket_order)
    feature_cols = resolve_feature_cols(args)

    df = load_dataset(args.dataset, feature_cols=feature_cols, label_col=args.label_col, bucket_order=bucket_order)
    use_ra_patches = has_ra_patch_columns(df)
    patch_manifest = load_patch_manifest(args.dataset_root)
    validate_split_contract(
        df,
        require_val=args.external_val_dataset is None or args.best_metric.startswith("val_"),
    )
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("No train rows found.")
    if args.external_val_dataset is None and val_df.empty:
        raise ValueError("No val rows found and no --external-val-dataset was provided.")
    if args.has_acc_head:
        if args.acc_label_col not in train_df.columns:
            raise ValueError(
                f"--has-acc-head requires '{args.acc_label_col}' in the training CSV. "
                "The auxiliary head should not be trained without yes/no accumulatability labels."
            )
        n_acc = int(train_df[args.acc_label_col].astype(str).str.lower().isin(["yes", "no"]).sum())
        if n_acc == 0:
            raise ValueError(
                f"--has-acc-head requires at least one yes/no label in '{args.acc_label_col}' "
                "for the train split."
            )
    stats_checkpoint = args.feature_stats_checkpoint or args.resume_checkpoint
    resume_payload = None
    if args.resume_checkpoint:
        resume_payload = torch.load(args.resume_checkpoint, map_location="cpu")
    if stats_checkpoint:
        stats_payload = resume_payload if stats_checkpoint == args.resume_checkpoint else torch.load(stats_checkpoint, map_location="cpu")
        feature_means = np.asarray(stats_payload["feature_means"], dtype=np.float32)
        feature_stds = np.asarray(stats_payload["feature_stds"], dtype=np.float32)
        if len(feature_means) != len(feature_cols) or len(feature_stds) != len(feature_cols):
            raise ValueError("Feature stats checkpoint does not match the requested feature columns.")
    else:
        feature_means, feature_stds = compute_feature_stats(train_df, feature_cols)
    np.save(out_dir / "feature_means.npy", feature_means)
    np.save(out_dir / "feature_stds.npy", feature_stds)

    train_ds = HybridRDTemporalTrainDataset(
        train_df,
        dataset_root=args.dataset_root,
        feature_means=feature_means,
        feature_stds=feature_stds,
        feature_cols=feature_cols,
        bucket_order=bucket_order,
        label_col=args.label_col,
        window_size=args.window_size,
        n_points=args.points_per_frame,
        seed=args.seed,
        sample_weight_col=args.sample_weight_col,
        acc_label_col=args.acc_label_col if args.has_acc_head else None,
        use_ra_patches=use_ra_patches,
    )
    class_counts = np.array([float((train_df[args.label_col] == b).sum()) for b in bucket_order], dtype=np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    train_sampler = None
    sampler_stats: dict[str, float] = {}
    if args.balanced_window_sampler:
        train_sampler, sampler_stats = build_balanced_window_sampler(
            train_ds,
            class_counts=class_counts,
            power=args.window_sampler_power,
            max_weight=args.window_sampler_max_weight,
        )
        print(f"balanced_window_sampler: {sampler_stats}", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.windows_per_batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )

    val_loader = None
    if not val_df.empty:
        val_ds = HybridRDTemporalEvalDataset(
            val_df,
            dataset_root=args.dataset_root,
            feature_means=feature_means,
            feature_stds=feature_stds,
            feature_cols=feature_cols,
            bucket_order=bucket_order,
            label_col=args.label_col,
            window_size=args.window_size,
            acc_label_col=args.acc_label_col if args.has_acc_head else None,
            use_ra_patches=use_ra_patches,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=hybrid_rd_eval_collate_fn)

    external_loader = None
    if args.external_val_dataset:
        ext_df = load_dataset(args.external_val_dataset, feature_cols=feature_cols, label_col=args.label_col, bucket_order=bucket_order)
        ext_ds = HybridRDTemporalEvalDataset(
            ext_df,
            dataset_root=args.external_val_root,
            feature_means=feature_means,
            feature_stds=feature_stds,
            feature_cols=feature_cols,
            bucket_order=bucket_order,
            label_col=args.label_col,
            window_size=args.window_size,
            acc_label_col=args.acc_label_col if args.has_acc_head else None,
            use_ra_patches=use_ra_patches,
        )
        external_loader = DataLoader(ext_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=hybrid_rd_eval_collate_fn)
    elif args.best_metric.startswith("external_"):
        raise ValueError(f"--best-metric {args.best_metric} requires --external-val-dataset.")

    patch_embed_dim = int(args.rd_patch_embed_dim)
    model_n_features = len(feature_cols) + patch_embed_dim * (2 if use_ra_patches else 1)
    base_model = KPConvTemporalSegmenter(
        n_features=model_n_features,
        n_classes=n_classes,
        window_size=args.window_size,
        k=args.k,
        emb_dims=args.emb_dims,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        dropout=args.dropout,
        frame_meta_dim=5,
        gate_hidden=args.gate_hidden,
        radius_1=args.radius_1,
        radius_2=args.radius_2,
        radius_3=args.radius_3,
        sigma_factor=args.sigma_factor,
        n_kernel_points=args.n_kernel_points,
    )
    model = RDPatchTemporalSegmenter(
        base_model,
        patch_channels=args.rd_patch_channels,
        patch_embed_dim=patch_embed_dim,
        use_ra_patches=use_ra_patches,
        has_acc_head=args.has_acc_head,
        acc_head_hidden_dim=args.acc_head_hidden_dim,
        patch_fusion_mode=args.patch_fusion_mode,
        rd_patch_doppler_bins=args.rd_patch_doppler_bins,
        rd_patch_range_bins=args.rd_patch_range_bins,
    ).to(device)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * n_classes
    weight_str = " ".join(f"{b}={w:.3f}" for b, w in zip(bucket_order, class_weights.tolist()))
    print(f"class_weights: {weight_str}", flush=True)
    if args.loss_kind == "focal":
        criterion = FocalLoss(weight=class_weights.to(device), gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="none")
    acc_criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    resume_epoch = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        if "optimizer_state" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if "scheduler_state" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        resume_epoch = int(resume_payload.get("epoch", 0))
        print(f"resumed checkpoint: {args.resume_checkpoint} at epoch {resume_epoch}", flush=True)

    config = vars(args).copy()
    config.update(
        {
            "encoder_type": "kpconv",
            "uses_rd_patch": True,
            "uses_ra_patch": bool(use_ra_patches),
            "patch_fusion_mode": args.patch_fusion_mode,
            "feature_cols": feature_cols,
            "bucket_order": bucket_order,
            "n_classes": n_classes,
            "device": str(device),
            "has_acc_head": bool(args.has_acc_head),
            "acc_label_col": args.acc_label_col if args.has_acc_head else None,
            "acc_loss_weight": float(args.acc_loss_weight if args.has_acc_head else 0.0),
            "balanced_window_sampler_stats": sampler_stats,
            "rd_patch_embed_dim": int(args.rd_patch_embed_dim),
            "ra_patch_embed_dim": int(args.rd_patch_embed_dim if use_ra_patches else 0),
            "rd_patch_frame_number_offset": int(patch_manifest.get("frame_number_offset", 0)),
            "ra_patch_source": str(patch_manifest.get("ra_patch_source", "true_ra")) if use_ra_patches else None,
            "ra_patch_az_mode": str(patch_manifest.get("ra_az_mode", "tx0_only")) if use_ra_patches else None,
            "ra_patch_az_fft_size": int(patch_manifest.get("ra_az_fft_size", 64)) if use_ra_patches else 0,
            "ra_patch_log_scale": bool(patch_manifest.get("ra_log_scale", True)) if use_ra_patches else False,
            "ra_patch_normalize_by_frame_max": bool(patch_manifest.get("ra_normalize_by_frame_max", True)) if use_ra_patches else False,
            "ra_patch_row_edge_mode": str(patch_manifest.get("ra_row_edge_mode", "zero_pad")) if use_ra_patches else None,
        }
    )
    (out_dir / "training_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    best_value = float(resume_payload.get("best_metric_value", float("nan"))) if resume_payload else float("nan")
    log_path = out_dir / "training_log.csv"
    if resume_payload is not None and log_path.exists():
        log_rows: list[dict[str, float]] = pd.read_csv(log_path).to_dict("records")
    else:
        log_rows = []

    def save_checkpoint(path: Path, epoch: int, extra: dict[str, object] | None = None) -> None:
        payload = {
            "encoder_type": "kpconv",
            "uses_rd_patch": True,
            "uses_ra_patch": bool(use_ra_patches),
            "patch_fusion_mode": args.patch_fusion_mode,
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "feature_means": feature_means.tolist(),
            "feature_stds": feature_stds.tolist(),
            "feature_cols": feature_cols,
            "bucket_order": bucket_order,
            "window_size": args.window_size,
            "points_per_frame": args.points_per_frame,
            "k": args.k,
            "emb_dims": args.emb_dims,
            "temporal_layers": args.temporal_layers,
            "temporal_heads": args.temporal_heads,
            "dropout": args.dropout,
            "frame_meta_dim": 5,
            "gate_hidden": args.gate_hidden,
            "radius_1": args.radius_1,
            "radius_2": args.radius_2,
            "radius_3": args.radius_3,
            "sigma_factor": args.sigma_factor,
            "n_kernel_points": args.n_kernel_points,
            "rd_patch_embed_dim": args.rd_patch_embed_dim,
            "rd_patch_channels": args.rd_patch_channels,
            "rd_patch_hidden_channels": [8, 16],
            "rd_patch_shape": [args.rd_patch_channels, args.rd_patch_doppler_bins, args.rd_patch_range_bins],
            "ra_patch_embed_dim": int(args.rd_patch_embed_dim if use_ra_patches else 0),
            "ra_patch_channels": int(args.rd_patch_channels if use_ra_patches else 0),
            "ra_patch_shape": [args.rd_patch_channels, 9, 7] if use_ra_patches else None,
            "rd_patch_frame_number_offset": int(patch_manifest.get("frame_number_offset", 0)),
            "ra_patch_source": str(patch_manifest.get("ra_patch_source", "true_ra")) if use_ra_patches else None,
            "ra_patch_az_mode": str(patch_manifest.get("ra_az_mode", "tx0_only")) if use_ra_patches else None,
            "ra_patch_az_fft_size": int(patch_manifest.get("ra_az_fft_size", 64)) if use_ra_patches else 0,
            "ra_patch_log_scale": bool(patch_manifest.get("ra_log_scale", True)) if use_ra_patches else False,
            "ra_patch_normalize_by_frame_max": bool(patch_manifest.get("ra_normalize_by_frame_max", True)) if use_ra_patches else False,
            "ra_patch_row_edge_mode": str(patch_manifest.get("ra_row_edge_mode", "zero_pad")) if use_ra_patches else None,
            "has_acc_head": bool(args.has_acc_head),
            "acc_label_col": args.acc_label_col if args.has_acc_head else None,
            "acc_loss_weight": float(args.acc_loss_weight if args.has_acc_head else 0.0),
            "acc_head_hidden_dim": int(args.acc_head_hidden_dim),
            "loss_kind": args.loss_kind,
            "focal_gamma": float(args.focal_gamma),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    for epoch in range(resume_epoch + 1, resume_epoch + args.epochs + 1):
        start = time.time()
        train_metrics = run_train_epoch(
            model, train_loader, criterion, acc_criterion, optimizer, device, n_classes, bucket_order,
            use_ra_patches=use_ra_patches,
            acc_loss_weight=args.acc_loss_weight if args.has_acc_head else 0.0,
        )
        scheduler.step()
        val_metrics = run_eval_epoch(
            model, val_loader, criterion, acc_criterion, device, n_classes, bucket_order,
            use_ra_patches=use_ra_patches,
            acc_loss_weight=args.acc_loss_weight if args.has_acc_head else 0.0,
        ) if val_loader else {}
        ext_metrics = run_eval_epoch(
            model, external_loader, criterion, acc_criterion, device, n_classes, bucket_order,
            use_ra_patches=use_ra_patches,
            acc_loss_weight=args.acc_loss_weight if args.has_acc_head else 0.0,
        ) if external_loader else {}
        row = {
            "epoch": float(epoch),
            "train_loss": metric_value(train_metrics, "loss"),
            "train_sem_loss": metric_value(train_metrics, "sem_loss"),
            "train_acc_loss": metric_value(train_metrics, "acc_loss"),
            "train_acc_auc": metric_value(train_metrics, "acc_auc"),
            "train_acc_precision": metric_value(train_metrics, "acc_precision"),
            "train_acc_recall": metric_value(train_metrics, "acc_recall"),
            "train_bal_acc": metric_value(train_metrics, "balanced_accuracy"),
            "val_loss": metric_value(val_metrics, "loss"),
            "val_sem_loss": metric_value(val_metrics, "sem_loss"),
            "val_acc_loss": metric_value(val_metrics, "acc_loss"),
            "val_acc_auc": metric_value(val_metrics, "acc_auc"),
            "val_acc_precision": metric_value(val_metrics, "acc_precision"),
            "val_acc_recall": metric_value(val_metrics, "acc_recall"),
            "val_bal_acc": metric_value(val_metrics, "balanced_accuracy"),
            "external_val_loss": metric_value(ext_metrics, "loss"),
            "external_val_acc_loss": metric_value(ext_metrics, "acc_loss"),
            "external_val_acc_auc": metric_value(ext_metrics, "acc_auc"),
            "external_val_acc_precision": metric_value(ext_metrics, "acc_precision"),
            "external_val_acc_recall": metric_value(ext_metrics, "acc_recall"),
            "external_val_bal_acc": metric_value(ext_metrics, "balanced_accuracy"),
            "epoch_sec": float(time.time() - start),
        }
        for prefix, m in [("train", train_metrics), ("val", val_metrics), ("ext", ext_metrics)]:
            cm = m.get("acc_confusion", {})
            if isinstance(cm, dict):
                for key in ("tn", "fp", "fn", "tp"):
                    row[f"{prefix}_acc_{key}"] = float(cm.get(key, float("nan")))
        # Flatten per-class metrics into row
        for prefix, m in [("train", train_metrics), ("val", val_metrics), ("ext", ext_metrics)]:
            for key in ("per_class_precision", "per_class_recall", "per_class_f1"):
                nested = m.get(key, {})
                for cls, val in nested.items():
                    row[f"{prefix}_{key[11:]}_{cls}"] = float(val)
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
        if args.has_acc_head:
            (out_dir / "accumulatability_validation_report.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val": {k: v for k, v in val_metrics.items() if str(k).startswith("acc_")},
                        "external_val": {k: v for k, v in ext_metrics.items() if str(k).startswith("acc_")},
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        current = row[args.best_metric]
        if better(args.best_metric, current, best_value):
            best_value = current
            save_checkpoint(out_dir / "best_model.pt", epoch, {"best_metric_value": best_value})
        save_checkpoint(out_dir / "last_model.pt", epoch, {"best_metric_value": best_value})
        current_lr = scheduler.get_last_lr()[0]

        # Build per-class recall print line
        pc_parts = []
        for prefix, m in [("train", train_metrics), ("val", val_metrics), ("ext", ext_metrics)]:
            rec = m.get("per_class_recall", {})
            cls_strs = [f"{cls[0]}r={rec.get(cls, float('nan')):.3f}" for cls in bucket_order if not np.isnan(rec.get(cls, float('nan')))]
            if cls_strs:
                pc_parts.append(f"{prefix}: [{' '.join(cls_strs)}]")
        pc_str = "  ".join(pc_parts)

        acc_part = f" val_acc_auc={row['val_acc_auc']:.4f}" if args.has_acc_head else ""
        print(
            f"epoch={epoch:03d} train_bal={row['train_bal_acc']:.4f} "
            f"val_bal={row['val_bal_acc']:.4f} ext_bal={row['external_val_bal_acc']:.4f} "
            f"{acc_part} "
            f"lr={current_lr:.2e}  {pc_str}",
            flush=True,
        )


if __name__ == "__main__":
    main()
