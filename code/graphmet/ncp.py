"""Neural Conditional Probability (NCP) inference on top of GraphMet.

GraphMet learns the leading Koopman singular triples (u_i, v_i, ς_i) by
construction. The NCP framework converts those into a closed-form
*conditional density* for the latent state at any lead time T:

    ℙ[z^{t+T} ∈ B | z^t = z]
        ≈ ∫_B (1 + Σ_i ς_i^T · u_i(z) · v_i(z')) μ(dz')

In practice we do not materialise the global Koopman matrix. We compute the
per-type SVD of each shared block K_α ∈ R^{m × m} and apply it
node-wise — this is mathematically consistent with the block-sparse
structure (each interaction type acts independently in its own block).

This module exposes:
  - `NCPPredictor.left_features(z)`  → u(z) :  [N_M, m]
  - `NCPPredictor.right_features(z)` → v(z) :  [N_M, m]
  - `NCPPredictor.transition_log_density(z_t, z_tp, T)`
  - `NCPPredictor.ensemble(z_t, T, n_samples)`  → analytic samples
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .koopman import BlockSparseKoopman


class NCPPredictor(nn.Module):
    """Wraps a trained Koopman processor and exposes NCP-style inference."""

    def __init__(self, koopman: BlockSparseKoopman):
        super().__init__()
        self.k = koopman
        U, S, V = koopman.block_svd()       # [k, m, m], [k, m], [k, m, m]
        self.register_buffer("U", U, persistent=False)
        self.register_buffer("S", S, persistent=False)
        self.register_buffer("V", V, persistent=False)

    # ── Singular features ─────────────────────────────────────────────

    def _per_type_project(self, z: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Project each node's latent vector through its type-specific basis.

        We pick the *self-loop* type as the canonical projection (every node
        has exactly one self-loop). For nodes whose self-loop type is k_t,
        the projection is basis[k_t] · z_node.
        """
        # Determine each node's self-loop type
        ei = self.k
        self_mask = (ei.edge_src == ei.edge_dst)
        node_to_type = torch.zeros(ei.n_nodes, dtype=torch.long, device=z.device)
        node_to_type[ei.edge_src[self_mask]] = ei.edge_type[self_mask]
        B_per_node = basis[node_to_type]      # [N_M, m, m]
        # einsum over node-wise matrix product
        return torch.einsum("nij,bnj->bni", B_per_node, z)

    def left_features(self, z: torch.Tensor) -> torch.Tensor:
        """u_i(z) — projection of z onto the left singular basis per node."""
        return self._per_type_project(z, self.U.transpose(-2, -1))

    def right_features(self, z: torch.Tensor) -> torch.Tensor:
        """v_i(z) — projection onto the right singular basis per node."""
        return self._per_type_project(z, self.V.transpose(-2, -1))

    # ── Transition density ───────────────────────────────────────────

    def transition_log_density(
        self,
        z_t: torch.Tensor,
        z_tp: torch.Tensor,
        lead_time: int = 1,
        rank: Optional[int] = None,
    ) -> torch.Tensor:
        """Log of the NCP density 1 + Σ ς^T u(z) v(z').

        For numerical safety we shift by log of the truncated kernel value,
        and clip the inner sum to keep the argument positive.
        """
        u = self.left_features(z_t)            # [B, N, m]
        v = self.right_features(z_tp)          # [B, N, m]
        # broadcast singular values to nodes via the self-loop type
        self_mask = (self.k.edge_src == self.k.edge_dst)
        node_to_type = torch.zeros(self.k.n_nodes, dtype=torch.long, device=z_t.device)
        node_to_type[self.k.edge_src[self_mask]] = self.k.edge_type[self_mask]
        sigma = self.S[node_to_type]           # [N, m]
        s_T = sigma.pow(float(lead_time))
        if rank is not None:
            s_T[..., rank:] = 0.0
        kernel = 1.0 + (s_T.unsqueeze(0) * u * v).sum(dim=-1)   # [B, N]
        return kernel.clamp_min(1e-6).log()

    # ── Analytic sampling ─────────────────────────────────────────────

    @torch.no_grad()
    def ensemble(
        self,
        z_t: torch.Tensor,
        lead_time: int,
        n_samples: int = 50,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate `n_samples` analytic ensemble realisations at lead T.

        This is the linear-Koopman analogue of the NCP sampler: the
        deterministic Koopman rollout K^T z plus structured noise drawn from
        the per-mode covariance diag(1 − ς_i^{2T}). For unitary modes
        (ς_i = 1) the variance is zero; for fast-decaying modes the
        variance saturates at 1.
        """
        # Deterministic rollout
        z = z_t.clone()
        for _ in range(lead_time):
            z = self.k(z)
        b, n, m = z.shape

        # Per-mode noise variance
        self_mask = (self.k.edge_src == self.k.edge_dst)
        node_to_type = torch.zeros(self.k.n_nodes, dtype=torch.long, device=z.device)
        node_to_type[self.k.edge_src[self_mask]] = self.k.edge_type[self_mask]
        sigma = self.S[node_to_type]                  # [N, m]
        var = (1.0 - sigma.pow(2.0 * float(lead_time))).clamp_min(0.0)
        std = var.sqrt() * float(noise_scale)        # [N, m]

        # Sample in singular basis, then map back: ξ = V_α · ε
        eps = torch.randn(n_samples, b, n, m, device=z.device, dtype=z.dtype)
        eps = eps * std.unsqueeze(0).unsqueeze(0)
        V_per_node = self.V[node_to_type]            # [N, m, m]
        xi = torch.einsum("nij,sbnj->sbni", V_per_node, eps)
        return z.unsqueeze(0) + xi                   # [S, B, N, m]
