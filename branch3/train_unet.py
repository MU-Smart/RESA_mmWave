"""Train the Branch 3 BEV U-Net against indexed radar/label pairs."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config.config import BRANCH3_UNET_MODULE


_unet_mod = importlib.import_module(BRANCH3_UNET_MODULE)
BEVUNet = _unet_mod.BEVUNet
rd_cube_to_input = _unet_mod.rd_cube_to_input
save_unet_checkpoint = _unet_mod.save_unet_checkpoint
load_unet_checkpoint = _unet_mod.load_unet_checkpoint


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = pred.reshape(pred.size(0), -1)
        target_f = target.reshape(target.size(0), -1)
        intersection = (pred_f * target_f).sum(dim=1)
        return 1.0 - (
            (2.0 * intersection + self.smooth)
            / (pred_f.sum(dim=1) + target_f.sum(dim=1) + self.smooth)
        ).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, lam_bce: float = 0.5, lam_dice: float = 0.5) -> None:
        super().__init__()
        self.lam_bce = lam_bce
        self.lam_dice = lam_dice
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lam_bce * self.bce(pred, target) + self.lam_dice * self.dice(pred, target)


class UNetRadarDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        label_h: int = 128,
        label_w: int = 128,
        in_channels: int = 24,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.label_h = label_h
        self.label_w = label_w
        self.in_channels = in_channels
        self._npz_cache: dict[str, object] = {}

    def _get_npz(self, session_dir: str):
        if session_dir not in self._npz_cache:
            npz_paths = list(Path(session_dir).glob("*_radar_tensors.npz"))
            if not npz_paths:
                raise FileNotFoundError(f"No _radar_tensors.npz in {session_dir}")
            self._npz_cache[session_dir] = np.load(str(npz_paths[0]), allow_pickle=False)
        return self._npz_cache[session_dir]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        session_dir = str(row["session_dir"])
        frame_index = int(row.get("rd_frame_index", int(row["radar_frame_num"]) - 1))
        label_path = str(row["label_path"])

        npz = self._get_npz(session_dir)
        rd_key = f"rd_{frame_index}"
        if rd_key not in npz:
            raise KeyError(
                f"Missing {rd_key} in {session_dir}; catalog radar_frame_num "
                "uses the one-based ADC convention while radar tensors use "
                "zero-based frame indices."
            )
        rd_cube = npz[rd_key]
        inp = rd_cube_to_input(rd_cube, use_prior=False).squeeze(0)

        label = np.load(label_path).astype(np.float32)
        if label.shape != (self.label_h, self.label_w):
            label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
            label_t = F.interpolate(label_t, size=(self.label_h, self.label_w), mode="nearest")
            label = label_t.squeeze(0).squeeze(0).numpy()

        return inp, torch.from_numpy(label)


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    p = (pred > threshold).float().reshape(-1)
    t = target.float().reshape(-1)
    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    tn = ((1 - p) * (1 - t)).sum().item()
    eps = 1e-6
    return {
        "accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
        "f1": float((2 * tp) / (2 * tp + fp + fn + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "far": float(fp / (fp + tn + eps)),
    }


def run_train_epoch(
    model: BEVUNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_pred: list[torch.Tensor] = []
    all_tgt: list[torch.Tensor] = []

    for inp, label in loader:
        inp = inp.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        pred = model(inp)
        loss = criterion(pred, label)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * inp.size(0)
        all_pred.append(pred.detach().cpu())
        all_tgt.append(label.detach().cpu())

    metrics = compute_metrics(torch.cat(all_pred), torch.cat(all_tgt))
    metrics["loss"] = float(total_loss / len(loader.dataset))
    return metrics


@torch.no_grad()
def run_eval_epoch(
    model: BEVUNet,
    loader: DataLoader | None,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    if loader is None or len(loader.dataset) == 0:
        return {"loss": float("nan"), "iou": float("nan")}

    model.eval()
    total_loss = 0.0
    all_pred: list[torch.Tensor] = []
    all_tgt: list[torch.Tensor] = []

    for inp, label in loader:
        inp = inp.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        pred = model(inp)
        loss = criterion(pred, label)
        total_loss += loss.item() * inp.size(0)
        all_pred.append(pred.cpu())
        all_tgt.append(label.cpu())

    metrics = compute_metrics(torch.cat(all_pred), torch.cat(all_tgt))
    metrics["loss"] = float(total_loss / len(loader.dataset))
    return metrics


def _load_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"].astype(str) == split].reset_index(drop=True)


def _best_metric_key(metric: str) -> str:
    return {
        "val_loss": "val_loss",
        "val_iou": "val_iou",
        "ext_val_loss": "ext_val_loss",
        "ext_val_iou": "ext_val_iou",
    }[metric]


def _is_better(metric_name: str, cur: float, best: float) -> bool:
    if np.isnan(cur):
        return False
    if np.isnan(best):
        return True
    return cur < best if metric_name.endswith("loss") else cur > best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Branch 3 BEVUNet free-space segmenter.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ext_val_dataset", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.000314)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lam_bce", type=float, default=0.5)
    parser.add_argument("--lam_dice", type=float, default=0.5)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--best_metric",
        default="ext_val_iou",
        choices=["val_loss", "val_iou", "ext_val_loss", "ext_val_iou"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.dataset, low_memory=False)
    train_df = _load_split(df, "train")
    val_df = _load_split(df, "val")
    if train_df.empty:
        raise RuntimeError("Training dataset has no rows with split='train'.")

    train_ds = UNetRadarDataset(train_df)
    val_ds = UNetRadarDataset(val_df) if not val_df.empty else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ext_loader = None
    if args.ext_val_dataset:
        ext_df = pd.read_csv(args.ext_val_dataset, low_memory=False)
        ext_ds = UNetRadarDataset(ext_df)
        ext_loader = DataLoader(ext_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BEVUNet(in_channels=24, base_ch=args.base_ch, out_h=128, out_w=128).to(device)
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        ckpt, resumed_model = load_unet_checkpoint(args.resume_checkpoint, map_location="cpu")
        if (
            int(ckpt.get("in_channels", 24)) != 24
            or int(ckpt.get("base_ch", args.base_ch)) != args.base_ch
            or int(ckpt.get("out_h", 128)) != 128
            or int(ckpt.get("out_w", 128)) != 128
        ):
            raise RuntimeError(
                "Resume checkpoint architecture does not match requested "
                f"in_channels=24, base_ch={args.base_ch}, out_h=128, out_w=128."
            )
        model.load_state_dict(resumed_model.state_dict())

    criterion = BCEDiceLoss(lam_bce=args.lam_bce, lam_dice=args.lam_dice)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.01,
    )

    config = {
        "branch": "3",
        "model": "BEVUNet",
        "in_channels": 24,
        "base_ch": args.base_ch,
        "out_h": 128,
        "out_w": 128,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lam_bce": args.lam_bce,
        "lam_dice": args.lam_dice,
        "best_metric": args.best_metric,
        "patience": args.patience,
        "n_params": model.n_params(),
        "device": str(device),
        "seed": args.seed,
    }
    (out_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log_fields = [
        "epoch",
        "train_loss",
        "train_iou",
        "val_loss",
        "val_iou",
        "ext_val_loss",
        "ext_val_iou",
        "lr",
        "epoch_sec",
    ]
    log_path = out_dir / "training_log.csv"
    log_path.write_text(",".join(log_fields) + "\n", encoding="utf-8")

    best_value = float("nan")
    patience_counter = 0
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"
    compat_best = out_dir / "unet_best_model.pt"
    compat_last = out_dir / "unet_last_model.pt"

    def save_checkpoint(path: Path, epoch: int, *, extra: dict[str, float] | None = None) -> None:
        save_unet_checkpoint(path, epoch, model, extra=extra)

    print("=" * 72)
    print("train_unet.py  [BEVUNet, Branch 3]")
    print(f"  Device   : {device}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Ext val  : {args.ext_val_dataset}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Train frames : {len(train_df):,}")
    print(f"  Val frames   : {len(val_df):,}")
    if args.ext_val_dataset:
        print(f"  Ext val dataset : {args.ext_val_dataset}")
    print(f"  Model params : {model.n_params():,}")
    print("-" * 72)

    epoch = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = run_eval_epoch(model, val_loader, criterion, device)
        ext_metrics = run_eval_epoch(model, ext_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss", float("nan")),
            "train_iou": train_metrics.get("iou", float("nan")),
            "val_loss": val_metrics.get("loss", float("nan")),
            "val_iou": val_metrics.get("iou", float("nan")),
            "ext_val_loss": ext_metrics.get("loss", float("nan")),
            "ext_val_iou": ext_metrics.get("iou", float("nan")),
            "lr": scheduler.get_last_lr()[0],
            "epoch_sec": time.time() - start,
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(",".join(str(row[field]) for field in log_fields) + "\n")

        metric_key = _best_metric_key(args.best_metric)
        cur_val = float(row[metric_key]) if row[metric_key] == row[metric_key] else float("nan")
        is_best = _is_better(args.best_metric, cur_val, best_value)
        if is_best:
            best_value = cur_val
            patience_counter = 0
            save_checkpoint(best_path, epoch, extra={"best_metric": args.best_metric, "best_value": best_value})
            save_checkpoint(compat_best, epoch, extra={"best_metric": args.best_metric, "best_value": best_value})
        else:
            patience_counter += 1

        print(
            f"  ep {epoch:3d}/{args.epochs}"
            f"  tr_loss={row['train_loss']:.4f}"
            f"  tr_iou={row['train_iou']:.3f}"
            f"  val_iou={row['val_iou']:.3f}"
            f"  ext_iou={row['ext_val_iou']:.3f}{'*' if is_best else ' '}"
            f"  {row['epoch_sec']:.1f}s",
            flush=True,
        )

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch}.", flush=True)
            break

    save_checkpoint(last_path, epoch)
    save_checkpoint(compat_last, epoch)
    print(f"\n  Best {args.best_metric}: {best_value:.5f}")
    print(f"  Best model : {best_path}")
    print(f"  Last model : {last_path}")


if __name__ == "__main__":
    main()
