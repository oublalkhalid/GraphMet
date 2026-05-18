# GraphMet — Reference Implementation

Anonymous PyTorch implementation of
**Learning Spectral Compositional Koopman Operators for Global-to-Regional
Weather Forecasting** (under double-blind review).

This directory implements the model described in §4 of the paper:

```
                          ┌──────────────────┐
   x^t  (lat/lon grid) ──►│  Grid2Mesh (G2M) │──► mesh features
                          └──────────────────┘
                                    │
                       ┌──── NodeEncoder (per-node MLP) ────┐
                       ▼                                    │
            z^t  ∈ ℝ^{N_M × m}                              │
                       │                                    │
                       ▼                                    │
            ┌─────────────────────────────┐                 │
            │  Block-Sparse Koopman step  │   14,336 params │
            │      z^{t+1} = K z^t        │                 │
            └─────────────────────────────┘                 │
                       │                                    │
                       ▼                                    │
            z^{t+1} ∈ ℝ^{N_M × m}                           │
                       │                                    │
                       └──── NodeDecoder (per-node MLP) ────┘
                                    │
                          ┌──────────────────┐
                          │  Mesh2Grid (M2G) │──► x̂^{t+1}
                          └──────────────────┘
```

Trained jointly with the three-term spectral objective

```
𝒥(φ, K) = − L_spec(φ)
          + ξ₁ · [R(Ĉ_φ(t)) + R(Ĉ_φ(t+1))]
          + ξ₂ · L_phys(φ, K)
```

---

## Layout

```
code/
├── graphmet/
│   ├── mesh.py       multi-scale icosahedral mesh + edge types
│   ├── encoder.py    Grid2Mesh + per-node encoder
│   ├── decoder.py    per-node decoder + Mesh2Grid
│   ├── koopman.py    block-sparse Koopman processor K
│   ├── losses.py     L_spec, R isotropy, L_phys, GraphMetObjective
│   ├── model.py      full GraphMet pipeline + autoregressive rollout
│   └── ncp.py        Neural Conditional Probability inference
├── data/
│   ├── era5.py       ERA5 one-step pairs (xarray; synthetic fallback)
│   └── cerra.py      CERRA regional pairs (xarray; synthetic fallback)
├── scripts/
│   ├── train_global.py     Stage 1 — global pretraining on ERA5
│   ├── train_regional.py   Stage 2 — regional fine-tune on CERRA
│   ├── eval.py             deterministic RMSE / ACC / Activity
│   └── ncp_inference.py    analytic probabilistic forecasting
├── configs/
│   ├── global_era5.yaml
│   └── regional_cerra.yaml
└── requirements.txt
```

---

## Installation

```bash
conda create -n graphmet python=3.10 && conda activate graphmet
pip install -r requirements.txt
```

Only `torch` and `numpy` are mandatory; `xarray`, `zarr` and `netCDF4` are
required for reading real ERA5 / CERRA. Without them every dataset falls
back to a deterministic synthetic generator so the pipeline can still be
smoke-tested end-to-end.

---

## Quick smoke test (no data required)

```bash
cd code
python -m scripts.train_global --data synthetic --refine 3 --bs 4 --steps 50 --log-every 10
```

This builds a small (642-node) multi-mesh, runs 50 optimisation steps on
synthetic data, and prints the full `GraphMet` loss decomposition.

---

## Stage 1 — Global pretraining (ERA5)

```bash
python -m scripts.train_global \
    --data /path/to/era5 \
    --refine 6 --m 32 \
    --bs 8 --steps 500000 \
    --xi1 1e-3 --xi2 1.0 --lr 1e-4 \
    --ckpt checkpoints/graphmet_global.pt
```

Logs report

* `total`     — value of `𝒥(φ, K)`
* `L_spec`    — Koopman predictability (closer to 1 is better)
* `iso`       — isotropy penalty (closer to 0 is better)
* `L_phys`    — weighted one-step MSE
* `λ⁺_min`    — smallest positive eigenvalue of `Ĉ_φ(t)` (main paper diagnostic)

---

## Stage 2 — Regional fine-tune (CERRA, 5 km)

```bash
python -m scripts.train_regional \
    --data /path/to/cerra \
    --pretrained checkpoints/graphmet_global.pt \
    --steps 50000 --bs 4 --lr 5e-5 \
    --ckpt checkpoints/graphmet_regional.pt
```

Encoder + Koopman matrix are frozen; only the decoder is fine-tuned.

---

## Evaluation

```bash
python -m scripts.eval \
    --data /path/to/era5 \
    --checkpoint checkpoints/graphmet_global.pt \
    --lead-times 24 72 120 240 360
```

---

## Probabilistic inference (NCP)

```bash
python -m scripts.ncp_inference \
    --data /path/to/era5 \
    --checkpoint checkpoints/graphmet_global.pt \
    --mode ensemble --lead 120 --n-members 50
```

NCP uses the per-type SVD of the trained Koopman blocks
(`koopman.block_svd()`) to draw analytic ensemble realisations — no
diffusion sampling, no MCMC.

---

## Notes for reviewers

* All code is anonymous; no author, institution or e-mail appears anywhere.
* The model is faithful to §3–§4 of the paper. Hyperparameters are exposed
  via CLI flags so reviewers can reproduce ablations (`--refine`, `--m`,
  `--xi1`, `--xi2`, etc.).
* Real ERA5 / CERRA loading requires a `xarray`-readable store; without it,
  the synthetic generators reproduce the *shape and Markov structure* of
  the data so reviewers can run the full pipeline locally.
