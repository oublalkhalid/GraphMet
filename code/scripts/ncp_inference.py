"""Analytic probabilistic inference with NCP (no diffusion sampling).

Loads a trained GraphMet checkpoint, wraps its Koopman processor in
`NCPPredictor`, and generates either:
  (a) analytic ensemble realisations in physical space, or
  (b) the log transition density between an initial state z_t and a
      candidate future state z_{t+T}.

Usage
-----
    python -m scripts.ncp_inference \
        --data /path/to/era5 \
        --checkpoint checkpoints/graphmet_global.pt \
        --mode ensemble --lead 120 --n-members 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import ERA5Pairs
from graphmet import GraphMet, NCPPredictor, build_multimesh
from graphmet.model import GraphMetConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GraphMet NCP inference")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--refine", type=int, default=5)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--mode", choices=("ensemble", "density"), default="ensemble")
    p.add_argument("--lead", type=int, default=120,
                   help="Lead time in hours (must be multiple of 6).")
    p.add_argument("--n-members", type=int, default=50)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = ERA5Pairs(args.data)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, drop_last=True)
    d_in = next(iter(loader))[0].shape[-1]

    mesh = build_multimesh(n_refine=args.refine)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = GraphMetConfig(grid_in=d_in, grid_out=d_in, **{
        k: v for k, v in ckpt.get("cfg", {}).items()
        if k in {"latent_dim", "mesh_feat", "hidden", "g2m_k", "m2g_k"}
    })
    model = GraphMet(cfg, mesh, ds.grid_latlon()).to(args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    ncp = NCPPredictor(model.koopman).to(args.device)
    n_steps = args.lead // ds.lead_hours

    with torch.no_grad():
        xt, xtp = next(iter(loader))
        xt = xt.to(args.device); xtp = xtp.to(args.device)
        zt = model.encode(xt)

        if args.mode == "ensemble":
            ens_z = ncp.ensemble(zt, lead_time=n_steps,
                                 n_samples=args.n_members,
                                 noise_scale=args.noise_scale)
            # Decode each member back to physical space
            s, b, n, m = ens_z.shape
            ens_x = torch.stack([
                model.decode(ens_z[i], xt) for i in range(s)
            ], dim=0)                                      # [S, B, N_G, d]
            mean = ens_x.mean(dim=0)
            spread = ens_x.std(dim=0)
            print(f"ensemble: members={s}  mean‖·‖={mean.norm().item():.3f}  "
                  f"spread‖·‖={spread.norm().item():.3f}")
        else:
            ztp = model.encode(xtp)
            log_p = ncp.transition_log_density(zt, ztp, lead_time=n_steps)
            print(f"log density per node:  mean={log_p.mean().item():.3f}  "
                  f"min={log_p.min().item():.3f}  max={log_p.max().item():.3f}")


if __name__ == "__main__":
    main()
