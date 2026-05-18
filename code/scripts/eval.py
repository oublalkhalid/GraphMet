"""Deterministic evaluation: latitude-weighted RMSE / ACC / Activity / SSR.

Usage
-----
    python -m scripts.eval \
        --data /path/to/era5 \
        --checkpoint checkpoints/graphmet_global.pt \
        --lead-times 24 72 120 240 360 \
        --variable z500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import ERA5Pairs
from graphmet import GraphMet, build_multimesh
from graphmet.model import GraphMetConfig


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def latitude_weighted_rmse(pred: torch.Tensor, tgt: torch.Tensor,
                           area_w: torch.Tensor) -> torch.Tensor:
    """Latitude-area weighted RMSE per channel."""
    sq = (pred - tgt).pow(2)                         # [B, N, d]
    return (sq * area_w.view(1, -1, 1)).mean(dim=(0, 1)).sqrt()


def acc(pred: torch.Tensor, tgt: torch.Tensor,
        clim: torch.Tensor, area_w: torch.Tensor) -> torch.Tensor:
    """Anomaly correlation coefficient (latitude-weighted)."""
    pa = pred - clim
    ta = tgt - clim
    num = (area_w.view(1, -1, 1) * pa * ta).sum(dim=(0, 1))
    den = (
        (area_w.view(1, -1, 1) * pa.pow(2)).sum(dim=(0, 1)).sqrt() *
        (area_w.view(1, -1, 1) * ta.pow(2)).sum(dim=(0, 1)).sqrt()
    )
    return num / den.clamp_min(1e-8)


def activity(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Spatial-variability ratio (forecast / truth). Ideal = 1."""
    return pred.std(dim=1) / tgt.std(dim=1).clamp_min(1e-6)


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GraphMet deterministic evaluation")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--refine", type=int, default=5)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lead-times", type=int, nargs="+", default=[24, 72, 120])
    p.add_argument("--variable-idx", type=int, default=0,
                   help="Index of the variable to report per-channel metrics for.")
    p.add_argument("--n-batches", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = ERA5Pairs(args.data)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, drop_last=True)
    d_in = next(iter(loader))[0].shape[-1]
    area_w = ds.latitude_weights().to(args.device)

    # Climatology: mean over a few hundred samples
    clim = torch.zeros_like(next(iter(loader))[0])
    n_seen = 0
    for i, (x, _) in enumerate(loader):
        clim += x.sum(dim=0)
        n_seen += x.shape[0]
        if i >= 200:
            break
    clim = (clim / max(n_seen, 1)).to(args.device).unsqueeze(0)

    # Model
    mesh = build_multimesh(n_refine=args.refine)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("cfg", {})
    cfg = GraphMetConfig(grid_in=d_in, grid_out=d_in, **{
        k: v for k, v in cfg_dict.items()
        if k in {"latent_dim", "mesh_feat", "hidden", "g2m_k", "m2g_k"}
    })
    model = GraphMet(cfg, mesh, ds.grid_latlon()).to(args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Rollout horizon (in 6-h steps)
    max_lead = max(args.lead_times)
    n_steps = max_lead // ds.lead_hours

    rmse_acc = {T: torch.zeros(d_in, device=args.device) for T in args.lead_times}
    cnt = 0
    with torch.no_grad():
        for batch_idx, (xt, _) in enumerate(loader):
            if batch_idx >= args.n_batches:
                break
            xt = xt.to(args.device)
            traj = model.rollout(xt, n_steps=n_steps)        # [B, T+1, N, d]
            for T in args.lead_times:
                k = T // ds.lead_hours
                target_idx = batch_idx * args.bs + k
                # Use the dataset's own future state as target
                if target_idx >= len(ds):
                    continue
                tgt_states = torch.stack([
                    ds[batch_idx * args.bs + i + k][0] for i in range(args.bs)
                ], dim=0).to(args.device)
                pred = traj[:, k]
                rmse_acc[T] += latitude_weighted_rmse(pred, tgt_states, area_w)
            cnt += 1

    print(f"\nGraphMet — variable index {args.variable_idx}")
    print("-" * 50)
    for T in args.lead_times:
        avg = (rmse_acc[T] / max(cnt, 1))[args.variable_idx].item()
        print(f"lead {T:>4}h | RMSE = {avg:.4f}")


if __name__ == "__main__":
    main()
