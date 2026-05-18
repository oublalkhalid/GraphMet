"""Stage-1 training: global pretraining of GraphMet on ERA5.

Usage
-----
    python -m scripts.train_global \
        --data /path/to/era5 \
        --refine 5 --m 32 --bs 8 --steps 200000 \
        --xi1 1e-3 --xi2 1.0 --lr 1e-4 \
        --ckpt checkpoints/graphmet_global.pt

Pass `--data synthetic` to do a smoke run on the synthetic generator
without any real data.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import ERA5Pairs
from graphmet import (
    GraphMet, GraphMetObjective, build_multimesh,
)
from graphmet.model import GraphMetConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GraphMet global pretraining (Stage 1)")
    p.add_argument("--data", type=str, required=True,
                   help="Path to ERA5 zarr/netCDF, or 'synthetic'.")
    p.add_argument("--refine", type=int, default=5,
                   help="Icosahedral refinements (paper uses 6).")
    p.add_argument("--m", "--latent-dim", dest="m", type=int, default=32,
                   help="Per-node latent sub-embedding dimension.")
    p.add_argument("--mesh-feat", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--xi1", type=float, default=1e-3)
    p.add_argument("--xi2", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt", type=str, default="checkpoints/graphmet_global.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Data ──────────────────────────────────────────────────────────
    ds = ERA5Pairs(args.data)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True,
                        num_workers=0, drop_last=True)
    grid_latlon = ds.grid_latlon()
    area_w = ds.latitude_weights().to(args.device)
    d_in = next(iter(loader))[0].shape[-1]

    # ── Mesh ─────────────────────────────────────────────────────────
    mesh = build_multimesh(n_refine=args.refine)
    print(f"[mesh] level {args.refine} → {mesh.n_nodes} nodes, "
          f"{mesh.edge_index.shape[1]} edges, {mesh.n_types} types")

    # ── Model ────────────────────────────────────────────────────────
    cfg = GraphMetConfig(
        grid_in=d_in, grid_out=d_in,
        latent_dim=args.m, mesh_feat=args.mesh_feat, hidden=args.hidden,
    )
    model = GraphMet(cfg, mesh, grid_latlon).to(args.device)
    print(f"[model] params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[model] Koopman block params: {model.koopman.n_parameters:,}")

    obj = GraphMetObjective(xi1=args.xi1, xi2=args.xi2).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr_min,
    )

    # ── Train ────────────────────────────────────────────────────────
    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

    while step < args.steps:
        try:
            xt, xtp = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xt, xtp = next(data_iter)

        xt = xt.to(args.device, non_blocking=True)
        xtp = xtp.to(args.device, non_blocking=True)

        x_hat, zt, ztp1 = model(xt, return_latents=True)
        report = obj(zt, ztp1, x_hat, xtp, area_weights=area_w)

        opt.zero_grad(set_to_none=True)
        report.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:>7} | total {report.total.item():+.4f} "
                  f"| L_spec {report.l_spec.item():.4f} "
                  f"| iso {report.iso.item():.4f} "
                  f"| L_phys {report.l_phys.item():.4f} "
                  f"| λ⁺_min {report.lambda_min_plus.item():.3e} "
                  f"| lr {sched.get_last_lr()[0]:.2e} "
                  f"| {step / max(dt, 1e-6):.1f} steps/s")

        if step % max(args.log_every * 10, 1000) == 0 or step == args.steps:
            torch.save({"model": model.state_dict(),
                        "cfg": cfg.__dict__, "step": step},
                       args.ckpt)
            print(f"[ckpt] saved → {args.ckpt}")


if __name__ == "__main__":
    main()
