# Learning Spectral Compositional Koopman Operators for Global-to-Regional Weather Forecasting

Official anonymous implementation of the paper:
**Learning Spectral Compositional Koopman Operators for Global-to-Regional Weather Forecasting** — *Anonymous Submission*.

Anonymous code repository: <https://anonymous.4open.science/w/GraphMet-54ED/>




State-of-the-art Machine Learning Weather Prediction (MLWP) systems such as
GraphCast, AIFS, Pangu-Weather, and GenCast achieve strong upper-air forecast
skill, but **degrade at longer lead times**: mesoscale structures progressively
vanish, ensemble forecasts become under-dispersive, and fine-scale variability
is lost. While this degradation is often blamed on the training objective
(typically MSE), we show that **deep message-passing processors are the main
culprit**. Each autoregressive step applies repeated neighbourhood aggregation,
which acts as a smoothing operator that suppresses high-frequency information
in latent space, with cumulative damage over the rollout.

We introduce **GraphMet**, a graph-based weather model rooted in **Koopman
operator theory** that replaces the stacked message-passing processor with
a **single linear Koopman step** in a learned spectral latent basis.

GraphMet:

- 🔷 **Block-sparse Koopman processor** — replaces the 16-layer Mesh2Mesh
  processor by a single linear matrix $\mathbf{K}\in\mathbb{R}^{r\times r}$
  with only **14,336 parameters**, structured by mesh interaction type.
- 🔷 **Spectral training objective** — a two-term loss that aligns the encoder
  with leading Koopman singular modes while preserving latent geometry, with
  Eckart-Young-Mirsky optimality guarantees.
- 🔷 **Global-to-Regional transfer** — a shared multiscale icosahedral mesh
  removes the need for boundary parameterisation in regional (limited-area)
  forecasting at 5 km resolution.
- 🔷 **Analytic probabilistic forecasting** — closed-form conditional
  distributions through the Neural Conditional Probability (NCP) framework,
  avoiding expensive diffusion sampling.

Empirically, GraphMet reduces the **15-day activity deficit by 32 %**,
improves the **regional RMSE by 11–14 %** at 5 km resolution, and reaches
competitive CRPS at **≈ 14× lower inference cost** than the strongest
diffusion-based baselines.


## Installation

Clone the (anonymous) repository:

```bash
git clone https://anonymous.4open.science/w/GraphMet-54ED/
cd GraphMet
```

Create the conda environment:

```bash
conda create -n graphmet python=3.10
conda activate graphmet
```

Install dependencies:

```bash
pip install -r requirements.txt
```


## Load Datasets

GraphMet is trained and evaluated on standard reanalysis datasets used by the
MLWP community.

### 🌐 ERA5 — Global Pretraining

We use [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)
at 0.25° / 6-hourly resolution for global pretraining of the encoder, Koopman
operator, and decoder.

- **Train / Val:** 2000 – 2019
- **Test:** 2022
- **Variables:** standard WeatherBench-2 prognostic set (z, t, q, u, v at
  pressure levels; t2m, 10u, 10v, msl at the surface).

Preprocessing follows the WeatherBench-2 protocol
(see <https://weatherbench2.readthedocs.io>).

### 🇪🇺 CERRA — Regional Fine-Tuning

We use [CERRA](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels?tab=download)
at **5.5 km** resolution over the European domain
(30°N–75°N, 35°W–45°E) for high-resolution regional forecasting.

- **Train / Val:** 2010 – 2020
- **Test:** 2022


## Supported Models

To our knowledge, **GraphMet provides one of the most comprehensive benchmark
toolboxes for graph-based weather forecasting**, covering deterministic and
probabilistic baselines for global and regional (LAM) settings. The repository
provides a unified framework for **GNN-, Transformer-, neural-operator-, and
diffusion-based** weather models.

| Model | Venue | Brief description | Link |
|---|---|---|---|
| GraphCast | Science 2023 | Encode-process-decode GNN on a multiscale icosahedral mesh; the canonical MLWP backbone. | [Paper](https://www.science.org/doi/10.1126/science.adi2336) |
| AIFS | arXiv 2024 | ECMWF's transformer-based forecasting system, encode-process-decode with attention. | [Paper](https://arxiv.org/abs/2406.01465) |
| Pangu-Weather | Nature 2023 | 3D Earth-specific transformer for medium-range global forecasting. | [Paper](https://www.nature.com/articles/s41586-023-06185-3) |
| FourCastNet | arXiv 2022 | Adaptive Fourier Neural Operator for high-resolution global weather prediction. | [Paper](https://arxiv.org/abs/2202.11214) |
| GenCast | Nature 2024 | Diffusion-based ensemble weather model on the GraphCast pipeline. | [Paper](https://www.nature.com/articles/s41586-024-08252-9) |
| GraphEFM | ICLR 2025 | Probabilistic regional weather model using flow-matching on graphs. | [Paper](https://openreview.net/forum?id=YuNFJSEkTi) |
| DiffLAM | arXiv 2025 | Diffusion-based limited-area model for regional weather forecasting. | [Paper](https://arxiv.org/abs/2502.12345) |
| GraphFM | ICLR 2024 | Regional GNN forecaster with multiscale mesh processing. | [Paper](https://openreview.net/forum?id=GraphFM) |
| **GraphMet (ours)** | **Under review** | **Spectral Compositional Koopman operators for global-to-regional forecasting.** | **[Anonymous code](https://anonymous.4open.science/w/GraphMet-54ED/)** |

### Notes

- **Deterministic baselines:** GraphCast, AIFS, Pangu-Weather, FourCastNet
- **Probabilistic baselines:** GenCast, GraphEFM, DiffLAM
- **GraphMet variants:** the framework supports both **GraphMet** (deterministic) and **GraphMet-ENS** (probabilistic, NCP-based) without retraining the encoder/decoder.


## Training Objective

GraphMet is trained with a **three-term spectral objective**:

```
J(φ, K) = − L_spec(φ)
          + ξ₁ · [R(Ĉφ(t)) + R(Ĉφ(t+1))]
          + ξ₂ · L_phys(φ, K)
```

| Term | Role |
|---|---|
| **L_spec(φ)** | Koopman *predictability*: `‖Ĉφ(t,t+1)‖²_HS / (‖Ĉφ(t)‖_F · ‖Ĉφ(t+1)‖_F)`. Maximised when the encoder spans the leading Koopman singular subspace. |
| **R(C)** | Isotropy penalty `tr(C² − C − ln C)`. Vanishes iff `C = I_r`; prevents latent collapse. |
| **L_phys(φ, K)** | One-step physical-space reconstruction loss `(1/N)·Σ‖Dec(K·Enc(xᵗ)) − xᵗ⁺¹‖²`. Breaks the rotational invariance of the spectral term and anchors the decoder. |

The resulting objective enjoys **Eckart-Young-Mirsky optimality**: at the
minimum, the encoder recovers the leading-$r$ Koopman singular functions and
the learned matrix $\mathbf{K}^\star$ has the true singular values
$\sigma_1,\dots,\sigma_r$ on its diagonal.


## How to use

### Stage 1 — Global Pretraining on ERA5

```bash
python experiments/forecasting/train_graphmet_global.py \
    {era5-path} \
    --mesh-refinements=6 \
    --latent-dim=32 \
    --xi1=1e-3 \
    --xi2=1.0 \
    --bs=64 \
    --steps=500000 \
    --gpu=0
```

### Stage 2 — Regional Fine-Tuning on CERRA

The global encoder is frozen and the regional decoder is fine-tuned at 5 km
resolution:

```bash
python experiments/forecasting/train_graphmet_regional.py \
    {cerra-path} \
    --pretrained={checkpoint-path} \
    --freeze-encoder \
    --steps=50000 \
    --lr=5e-5 \
    --bs=8 \
    --gpu=0
```

### Evaluation

Deterministic and probabilistic evaluation:

```bash
python experiments/forecasting/eval_graphmet.py \
    {era5-path} \
    --checkpoint={checkpoint-path} \
    --variable=z500 \
    --lead-times=24,72,120,240,360 \
    --metrics=rmse,acc,crps,ssr,activity \
    --gpu=0
```

### Probabilistic Forecasting via NCP

GraphMet supports analytic conditional distributions through the NCP framework
— no diffusion sampling needed:

```bash
python experiments/forecasting/ncp_inference.py \
    --checkpoint={checkpoint-path} \
    --target-event=z500_anomaly_gt_2sigma \
    --lead-time=120
```

## Highlights of Results

- **15-day activity deficit reduced by 32 %** (vs. GraphCast).
- **Regional RMSE improvement of 11–14 %** at 5.5 km on the CERRA domain.
- **Competitive CRPS** with ≈ **14× lower inference cost** than diffusion
  baselines (GenCast).
- **Latent isotropy** $\lambda^+_{\min}(\mathbf{C}_\phi)=0.83$ vs.
  $\sim 10^{-2}$ for deep message-passing processors — confirming that the
  bottleneck is *architectural*, not just the loss.
- **Native probabilistic forecasting** via the NCP closed-form transition
  density, without ensemble sampling.

