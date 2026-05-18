"""Stage-2 fine-tuning: regional decoder on CERRA (5 km).

The global encoder + Koopman processor (Stage-1 checkpoint) are frozen and
only the decoder is fine-tuned on the CERRA domain. This matches the
"global-to-regional transfer" experimental protocol of the paper.

Usage
-----
    python -m scripts.train_regional \
        --data /path/to/cerra \
        --pretrained checkpoints/graphmet_global.pt \
        --steps 50000 --bs 4 --lr 5e-5 \
        --ckpt checkpoints/graphmet_regional.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import CERRAPairs
from graphmet import GraphMet, GraphMetObjective, build_multimesh
from graphmet.model import GraphMetConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GraphMet regional fine-tuning (Stage 2)")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--pretrained", type=str, required=True,
                   help="Path to Stage-1 ERA5 checkpoint.")
    p.add_argument("--refine", type=int, default=5)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--xi1", type=float, default=1e-3)
    p.add_argument("--xi2", type=float, default=1.0)
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt", type=str, default="checkpoints/graphmet_regional.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Data ─────────────────────────────────────────────────────────
    ds = CERRAPairs(args.data)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True,
                        num_workers=0, drop_last=True)
    grid_latlon = ds.grid_latlon()
    area_w = ds.latitude_weights().to(args.device)
    d_in = next(iter(loader))[0].shape[-1]

    # ── Mesh + model ────────────────────────────────────────────────
    mesh = build_multimesh(n_refine=args.refine)
    ckpt = torch.load(args.pretrained, map_location="cpu")
    cfg_dict = ckpt.get("cfg", {})
    cfg = GraphMetConfig(grid_in=d_in, grid_out=d_in, **{
        k: v for k, v in cfg_dict.items()
        if k in {"latent_dim", "mesh_feat", "hidden", "g2m_k", "m2g_k"}
    })
    model = GraphMet(cfg, mesh, grid_latlon).to(args.device)

    # Load Stage-1 weights; allow shape mismatches in the decoder I/O layer
    # because the regional grid has a different number of variables.
    state = ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if not (name.startswith("node_dec") or name.startswith("m2g")):
                param.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] trainable params: {sum(p.numel() for p in trainable):,}")

    obj = GraphMetObjective(xi1=args.xi1, xi2=args.xi2).to(args.device)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr_min)

    # ── Train ────────────────────────────────────────────────────────
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            xt, xtp = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xt, xtp = next(data_iter)
        xt = xt.to(args.device); xtp = xtp.to(args.device)
        x_hat, zt, ztp1 = model(xt, return_latents=True)
        report = obj(zt, ztp1, x_hat, xtp, area_weights=area_w)
        opt.zero_grad(set_to_none=True)
        report.total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step()
        step += 1
        if step % args.log_every == 0:
            print(f"step {step:>6} | total {report.total.item():+.4f} "
                  f"| L_phys {report.l_phys.item():.4f} "
                  f"| λ⁺_min {report.lambda_min_plus.item():.3e}")
        if step % max(args.log_every * 10, 1000) == 0 or step == args.steps:
            torch.save({"model": model.state_dict(),
                        "cfg": cfg.__dict__, "step": step}, args.ckpt)
            print(f"[ckpt] saved → {args.ckpt}")


if __name__ == "__main__":
    main()
