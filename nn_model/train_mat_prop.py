#!/usr/bin/env python3
"""
Train an MLP regressor on BiNE material embeddings to predict a chosen target:
- formation energy per atom (fe)
- band gap (bandgap)

Usage
-----
python matprop_train.py --target fe ...   # formation energy
python matprop_train.py --target bandgap ...  # band gap
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split


# -----------------------------
# Utilities
# -----------------------------

_TENSOR_RE = re.compile(r"tensor\(([-+0-9.eE]+)")

def parse_tensor_value(token: str) -> Optional[float]:
    token = token.strip()
    if not token:
        return None
    if token.startswith("tensor("):
        m = _TENSOR_RE.search(token)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None
    try:
        return float(token)
    except ValueError:
        return None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Data loading
# -----------------------------

def load_bine_embeddings(emb_file: str) -> Dict[str, np.ndarray]:
    emb: Dict[str, np.ndarray] = {}
    with open(emb_file, "r") as f:
        for line in f:
            values = line.split()
            if not values:
                continue
            # kept from your notebook for compatibility
            key = values[0][1:].split("__")[0]

            vec_vals: List[float] = []
            for tok in values[1:]:
                v = parse_tensor_value(tok)
                if v is not None:
                    vec_vals.append(v)

            if len(vec_vals) == 0:
                continue
            emb[key] = np.asarray(vec_vals, dtype=np.float32)
    return emb


def load_features_pkl(features_pkl: str) -> dict:
    import pickle
    with open(features_pkl, "rb") as f:
        return pickle.load(f)


def build_dataset_for_target(
    emb: Dict[str, np.ndarray],
    features: dict,
    *,
    target: str,
    fe_min: float = -8.0,
    bg_min: float = 0.0,
    bg_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    target: "fe" or "bandgap"
    - fe uses key "formation_energy_per_atom" and filter fe >= fe_min
    - bandgap uses key "band_gap" and filter bg in [bg_min, bg_max] (bg_max optional)
    """
    if target == "fe":
        key = "formation_energy_per_atom"
    elif target == "bandgap":
        key = "band_gap"
    else:
        raise ValueError("target must be one of: fe, bandgap")

    ids: List[str] = []
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for mid, vec in emb.items():
        if mid not in features:
            continue
        v = features[mid].get(key, None)
        if v is None:
            continue

        try:
            y = float(v)
        except Exception:
            continue

        # filtering
        if target == "fe":
            if y < fe_min:
                continue
        else:
            # bandgap
            if y < bg_min:
                continue
            if bg_max is not None and y > bg_max:
                continue

        if not np.all(np.isfinite(vec)):
            continue

        ids.append(mid)
        X_list.append(vec.astype(np.float32, copy=False))
        y_list.append(np.float32(y))

    if len(X_list) == 0:
        raise ValueError("No samples left after filtering. Check your inputs/filters.")

    X = np.vstack([x[None, :] for x in X_list]).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y, ids


# -----------------------------
# Dataset / Model
# -----------------------------

class MaterialDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MaterialMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Metrics
# -----------------------------

@torch.no_grad()
def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_mse_sum = 0.0
    total_abs_sum = 0.0
    n = 0

    preds_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []

    mse_loss = nn.MSELoss(reduction="mean")

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(-1)
        out = model(x)

        mse = mse_loss(out, y)
        total_mse_sum += float(mse.item()) * x.size(0)
        total_abs_sum += float(torch.abs(out - y).sum().item())
        n += x.size(0)

        preds_all.append(out.detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds_all, axis=0).squeeze()
    targets = np.concatenate(targets_all, axis=0).squeeze()

    mse = total_mse_sum / max(n, 1)
    mae = total_abs_sum / max(n, 1)
    rmse = float(np.sqrt(mse))

    denom = float(np.sum((targets - targets.mean()) ** 2))
    r2 = float("nan") if denom <= 0 else float(1.0 - np.sum((targets - preds) ** 2) / denom)

    return {"mse": float(mse), "mae": float(mae), "rmse": rmse, "r2": r2, "preds": preds, "targets": targets}


# -----------------------------
# Training
# -----------------------------

@dataclass
class RunConfig:
    target: str
    emb_file: str
    features_pkl: str
    save_dir: str
    epochs: int
    batch_size: int
    lr: float
    hidden_dim: int
    split_train: float
    split_val: float
    split_test: float
    seed: int
    num_workers: int
    shuffle_train: bool
    weight_decay: float
    save_best_metric: str  # "val_mae" or "val_mse"
    device: str


def save_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_score: float, cfg: RunConfig) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "config": asdict(cfg),
        },
        path,
    )


def train_one_run(dataset: MaterialDataset, cfg: RunConfig, run_name: str, plots: bool, clamp_nonneg: bool, plot_range: Optional[Tuple[float, float]]) -> dict:
    import pandas as pd
    import matplotlib.pyplot as plt

    ensure_dir(Path(cfg.save_dir))
    run_dir = Path(cfg.save_dir) / run_name
    ensure_dir(run_dir)

    # split
    n_total = len(dataset)
    n_train = int(cfg.split_train * n_total)
    n_val = int(cfg.split_val * n_total)
    n_test = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=cfg.shuffle_train, num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)

    input_dim = int(dataset.X.shape[1])
    model = MaterialMLP(input_dim=input_dim, hidden_dim=cfg.hidden_dim, output_dim=1)

    device = torch.device(cfg.device)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"epoch": [], "train_mse": [], "val_mse": [], "val_mae": [], "lr": []}
    best_score = float("inf")
    best_epoch = -1
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_sum = 0.0
        n_seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_sum += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        train_mse = train_sum / max(n_seen, 1)
        val_metrics = evaluate_regression(model, val_loader, device)
        val_mse = float(val_metrics["mse"])
        val_mae = float(val_metrics["mae"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        history["lr"].append(lr_now)

        score = val_mae if cfg.save_best_metric == "val_mae" else val_mse
        if score < best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(best_path, model, optimizer, epoch, best_score, cfg)

        save_checkpoint(last_path, model, optimizer, epoch, best_score, cfg)

    # reload best and test
    best_ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)

    val_best = evaluate_regression(model, val_loader, device)
    test_best = evaluate_regression(model, test_loader, device)

    # optional clamp for bandgap parity plot
    preds_plot = test_best["preds"].copy()
    targs_plot = test_best["targets"].copy()
    if clamp_nonneg:
        preds_plot = np.maximum(preds_plot, 0.0)
        targs_plot = np.maximum(targs_plot, 0.0)

    # save history + summary
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    summary = {
        "run_name": run_name,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "config": asdict(cfg),
        "val": {k: float(val_best[k]) for k in ["mse", "mae", "rmse", "r2"]},
        "test": {k: float(test_best[k]) for k in ["mse", "mae", "rmse", "r2"]},
        "paths": {"best_ckpt": str(best_path), "last_ckpt": str(last_path), "history_csv": str(run_dir / "history.csv")},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if plots:
        # loss curves
        plt.figure(figsize=(6, 6))
        plt.plot(history["epoch"], history["train_mse"], label="Train MSE")
        plt.plot(history["epoch"], history["val_mse"], label="Val MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "loss_curves.png", dpi=200)
        plt.close()

        # parity
        if plot_range is None:
            p1 = float(max(np.max(targs_plot), np.max(preds_plot)))
            p2 = float(min(np.min(targs_plot), np.min(preds_plot)))
        else:
            p2, p1 = float(plot_range[0]), float(plot_range[1])

        plt.figure(figsize=(6, 6))
        plt.scatter(preds_plot, targs_plot, s=6)
        plt.plot([p2, p1], [p2, p1])
        plt.axis("equal")
        plt.xlim(p2, p1)
        plt.ylim(p2, p1)
        xlabel = "Predicted FE (eV/atom)" if cfg.target == "fe" else "Predicted band gap (eV)"
        ylabel = "True FE (eV/atom)" if cfg.target == "fe" else "True band gap (eV)"
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{run_name} | test MAE={test_best['mae']:.3f}")
        plt.tight_layout()
        plt.savefig(run_dir / "pred_test.png", dpi=200)
        plt.close()

    return summary


def run_grid(
    *,
    target: str,
    emb_file: str,
    features_pkl: str,
    save_dir: str,
    epochs: int,
    batch_sizes: List[int],
    lrs: List[float],
    hidden_dim: int,
    split: Tuple[float, float, float],
    seed: int,
    num_workers: int,
    shuffle_train: bool,
    weight_decay: float,
    save_best_metric: str,
    fe_min: float,
    bg_min: float,
    bg_max: Optional[float],
    plots: bool,
    device: Optional[str],
    parity_range: Optional[Tuple[float, float]],
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    emb = load_bine_embeddings(emb_file)
    features = load_features_pkl(features_pkl)

    X, y, _ids = build_dataset_for_target(
        emb,
        features,
        target=target,
        fe_min=fe_min,
        bg_min=bg_min,
        bg_max=bg_max,
    )
    dataset = MaterialDataset(X, y)

    ensure_dir(Path(save_dir))
    all_summaries: List[dict] = []
    best_overall = None
    best_score = float("inf")

    # plotting conventions
    clamp_nonneg = (target == "bandgap")
    plot_range = parity_range
    if plot_range is None and target == "bandgap":
        # mimic your notebook default p1,p2 = 8,0
        plot_range = (0.0, 8.0)

    for bs in batch_sizes:
        for lr in lrs:
            cfg = RunConfig(
                target=target,
                emb_file=emb_file,
                features_pkl=features_pkl,
                save_dir=save_dir,
                epochs=epochs,
                batch_size=bs,
                lr=lr,
                hidden_dim=hidden_dim,
                split_train=split[0],
                split_val=split[1],
                split_test=split[2],
                seed=seed,
                num_workers=num_workers,
                shuffle_train=shuffle_train,
                weight_decay=weight_decay,
                save_best_metric=save_best_metric,
                device=device,
            )
            run_name = f"{target}_bs{bs}_lr{lr:g}_hid{hidden_dim}_seed{seed}"
            summary = train_one_run(dataset, cfg, run_name, plots=plots, clamp_nonneg=clamp_nonneg, plot_range=plot_range)
            all_summaries.append(summary)

            score = summary["val"]["mae"] if save_best_metric == "val_mae" else summary["val"]["mse"]
            if score < best_score:
                best_score = score
                best_overall = summary

    Path(save_dir, "grid_index.json").write_text(json.dumps({"best_overall": best_overall, "all_runs": all_summaries}, indent=2))
    return best_overall


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train FE or band gap regressor on BiNE embeddings.")
    p.add_argument("--target", required=True, choices=["fe", "bandgap"], help="Which property to train on.")
    p.add_argument("--emb-file", required=True, type=str)
    p.add_argument("--features-pkl", required=True, type=str)
    p.add_argument("--save-dir", default="outputs/matprop_runs", type=str)

    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 32, 16, 8])
    p.add_argument("--lrs", nargs="+", type=float, default=[0.01, 0.005, 0.001])
    p.add_argument("--hidden-dim", default=32, type=int)

    p.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    p.add_argument("--seed", default=7, type=int)
    p.add_argument("--num-workers", default=0, type=int)
    p.add_argument("--no-shuffle-train", action="store_true")
    p.add_argument("--weight-decay", default=0.0, type=float)
    p.add_argument("--save-best-metric", default="val_mae", choices=["val_mae", "val_mse"])

    # target-specific filters
    p.add_argument("--fe-min", default=-8.0, type=float, help="Keep FE >= fe-min.")
    p.add_argument("--bg-min", default=0.0, type=float, help="Keep band gap >= bg-min.")
    p.add_argument("--bg-max", default=None, type=float, help="Optional: keep band gap <= bg-max.")

    p.add_argument("--device", default=None, type=str, help="cuda/cpu (auto if omitted).")
    p.add_argument("--plots", action="store_true", help="Save loss and parity plots.")
    p.add_argument("--parity-range", nargs=2, type=float, default=None,
                   help="Optional parity plot range: min max. For bandgap defaults to 0 8.")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    split = tuple(args.split)
    if not np.isclose(sum(split), 1.0, atol=1e-3):
        raise SystemExit(f"--split must sum to 1.0 (got {split}, sum={sum(split):.6f})")

    set_seed(args.seed)

    best = run_grid(
        target=args.target,
        emb_file=args.emb_file,
        features_pkl=args.features_pkl,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_sizes=args.batch_sizes,
        lrs=args.lrs,
        hidden_dim=args.hidden_dim,
        split=split,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle_train=(not args.no_shuffle_train),
        weight_decay=args.weight_decay,
        save_best_metric=args.save_best_metric,
        fe_min=args.fe_min,
        bg_min=args.bg_min,
        bg_max=args.bg_max,
        plots=args.plots,
        device=args.device,
        parity_range=None if args.parity_range is None else (args.parity_range[0], args.parity_range[1]),
    )

    print("\n=== Best run (by {}) ===".format(args.save_best_metric))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
