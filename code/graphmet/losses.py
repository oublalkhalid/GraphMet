"""GraphMet spectral training objective.

Implements the three-term loss

    𝒥(φ, K) = − L_spec(φ)
              + ξ₁ · [R(Ĉ_φ(t)) + R(Ĉ_φ(t+1))]
              + ξ₂ · L_phys(φ, K)

where:

* L_spec(φ)  : Koopman predictability
               ‖Ĉ_φ(t, t+1)‖²_HS / (‖Ĉ_φ(t)‖_F · ‖Ĉ_φ(t+1)‖_F)
               in [0, 1]; maximised when the encoder spans the leading
               singular subspace of the true Koopman operator.

* R(C)       : isotropy penalty  tr(C² − C − ln C)  ≥ 0,
               with equality iff C = I_r. Prevents latent collapse.

* L_phys     : (weighted) one-step reconstruction loss
               (1/N) Σ ‖Dec(K · Enc(x^t)) − x^{t+1}‖²_w
               anchors the decoder in physical space and breaks the
               rotational ambiguity left by the spectral term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────────────
# Empirical covariances
# ────────────────────────────────────────────────────────────────────────────

def _flatten_batch(z: torch.Tensor) -> torch.Tensor:
    """Reshape latent z [B, N, m] → [B, r] with r = N·m."""
    if z.dim() == 3:
        b, n, m = z.shape
        return z.reshape(b, n * m)
    if z.dim() == 2:
        return z
    raise ValueError(f"unexpected latent rank {z.dim()}")


def empirical_covariance(z: torch.Tensor) -> torch.Tensor:
    """Empirical covariance Ĉ = (1/B) Z^T Z over the batch dimension.

    Returns a [r, r] tensor where r = N · m (latent dimension).
    """
    zf = _flatten_batch(z)
    b = zf.shape[0]
    return (zf.transpose(0, 1) @ zf) / max(b, 1)


def empirical_cross_covariance(zt: torch.Tensor, ztp1: torch.Tensor) -> torch.Tensor:
    """Empirical cross-covariance Ĉ(t, t+1) = (1/B) Z(t)^T Z(t+1)."""
    a = _flatten_batch(zt)
    b = _flatten_batch(ztp1)
    n = a.shape[0]
    return (a.transpose(0, 1) @ b) / max(n, 1)


# ────────────────────────────────────────────────────────────────────────────
# Spectral predictability  L_spec
# ────────────────────────────────────────────────────────────────────────────

def spectral_predictability(
    zt: torch.Tensor,
    ztp1: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scalar in [0, 1]; higher = more Koopman-predictable in this basis.

    Equation (5) of the paper:
        L_spec = ‖Ĉ(t, t+1)‖²_HS  /  (‖Ĉ(t)‖_F · ‖Ĉ(t+1)‖_F).

    We compute it without ever materialising r×r matrices when r is huge by
    using the small-side covariance Z^T Z (size B×B) when B < r. The current
    implementation forms the r×r covariance directly — fine for small mesh
    refinements; rely on the batch-side trick if you scale up r.
    """
    C_xx  = empirical_covariance(zt)
    C_yy  = empirical_covariance(ztp1)
    C_xy  = empirical_cross_covariance(zt, ztp1)
    num = (C_xy * C_xy).sum()                              # ‖·‖²_HS
    den = torch.linalg.matrix_norm(C_xx, ord="fro") * \
          torch.linalg.matrix_norm(C_yy, ord="fro")
    return num / den.clamp_min(eps)


# ────────────────────────────────────────────────────────────────────────────
# Isotropy penalty  R(C) = tr(C² − C − ln C)
# ────────────────────────────────────────────────────────────────────────────

def isotropy_penalty(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """R(C) ≥ 0 with R(I_r) = 0. Vectorised via the eigenvalues of C.

    For numerical safety the log is taken on max(λ_i, eps); training
    pushes eigenvalues towards 1 so the floor is never active at convergence.
    """
    C = empirical_covariance(z)
    # symmetrise to kill rounding noise then take real eigenvalues
    C = 0.5 * (C + C.transpose(-2, -1))
    eigvals = torch.linalg.eigvalsh(C)                     # [r]
    eigvals = eigvals.clamp_min(eps)
    return (eigvals.pow(2) - eigvals - eigvals.log()).sum()


# ────────────────────────────────────────────────────────────────────────────
# Physical-space one-step loss  L_phys
# ────────────────────────────────────────────────────────────────────────────

def physical_loss(
    x_pred: torch.Tensor,            # [B, N_G, d]
    x_true: torch.Tensor,            # [B, N_G, d]
    area_weights: Optional[torch.Tensor] = None,   # [N_G]
    variable_weights: Optional[torch.Tensor] = None,  # [d]
) -> torch.Tensor:
    """Latitude- and variable-weighted MSE on the physical grid."""
    diff = (x_pred - x_true).pow(2)                        # [B, N_G, d]
    if variable_weights is not None:
        diff = diff * variable_weights.view(1, 1, -1)
    if area_weights is not None:
        diff = diff * area_weights.view(1, -1, 1)
    return diff.mean()


# ────────────────────────────────────────────────────────────────────────────
# Combined objective
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphMetLossReport:
    total: torch.Tensor
    l_spec: torch.Tensor
    iso: torch.Tensor
    l_phys: torch.Tensor
    lambda_min_plus: torch.Tensor


class GraphMetObjective(nn.Module):
    """Combined GraphMet training objective.

    `xi1`, `xi2` are the loss weights (paper uses ξ₁ ≈ 1e-3, ξ₂ ≈ 1.0).
    """

    def __init__(self, xi1: float = 1e-3, xi2: float = 1.0):
        super().__init__()
        self.xi1 = float(xi1)
        self.xi2 = float(xi2)

    def forward(
        self,
        zt: torch.Tensor,
        ztp1: torch.Tensor,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        area_weights: Optional[torch.Tensor] = None,
        variable_weights: Optional[torch.Tensor] = None,
    ) -> GraphMetLossReport:
        l_spec = spectral_predictability(zt, ztp1)
        iso = isotropy_penalty(zt) + isotropy_penalty(ztp1)
        l_phys = physical_loss(x_pred, x_true, area_weights, variable_weights)
        total = -l_spec + self.xi1 * iso + self.xi2 * l_phys

        # λ⁺_min diagnostic (paper's main ablation metric)
        with torch.no_grad():
            C = empirical_covariance(zt)
            C = 0.5 * (C + C.transpose(-2, -1))
            eigs = torch.linalg.eigvalsh(C).clamp_min(0.0)
            pos = eigs[eigs > 1e-12]
            lam_min = pos.min() if pos.numel() > 0 else eigs.min()

        return GraphMetLossReport(
            total=total, l_spec=l_spec.detach(),
            iso=iso.detach(), l_phys=l_phys.detach(),
            lambda_min_plus=lam_min.detach(),
        )
