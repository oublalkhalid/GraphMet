"""Block-sparse Koopman processor.

Implements the single linear step

    z_i^{t+1} = Σ_{(j,i) ∈ ℰ_M}  K_{α_{ij}}  z_j^t

where K_α ∈ R^{m × m} are k = n_types learnable sub-matrices shared across
all edges of the same interaction type. With m = 32 and k = 14, the entire
processor has only 14 × 32² = 14,336 parameters.

This is the *one* architectural change relative to GraphCast: a 16-layer
message-passing block is replaced by this single matrix-vector multiply
applied once per autoregressive step.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BlockSparseKoopman(nn.Module):
    """Block-sparse Koopman matrix K with `n_types` shared sub-matrices.

    Parameters
    ----------
    latent_dim : int
        Per-node sub-embedding dimension m (paper uses 32).
    n_types : int
        Number of distinct edge interaction types (paper uses 14).
    edge_index : LongTensor [2, E]
        Directed edges (j → i) on the multi-mesh.
    edge_type : LongTensor [E]
        Class index in [0, n_types) for each edge.
    init_scale : float
        Init multiplier for the K_α blocks. The default 1/√m initialisation
        keeps E[‖K z‖²] ≈ ‖z‖² when the input is unit-variance, which is
        important to avoid latent collapse at the start of training.
    """

    def __init__(
        self,
        latent_dim: int,
        n_types: int,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        init_scale: Optional[float] = None,
    ):
        super().__init__()
        self.m = int(latent_dim)
        self.k = int(n_types)
        if init_scale is None:
            init_scale = 1.0 / math.sqrt(self.m)

        # K = stack of K_α matrices, shape [k, m, m]
        K = torch.randn(self.k, self.m, self.m) * init_scale
        self.K = nn.Parameter(K)

        # Register edges as buffers (non-trainable, moved with the module)
        self.register_buffer("edge_src", edge_index[0].long(), persistent=False)
        self.register_buffer("edge_dst", edge_index[1].long(), persistent=False)
        self.register_buffer("edge_type", edge_type.long(), persistent=False)

        # Per-destination normaliser (in-degree). The aggregation is a *sum*
        # over neighbours, but we precompute the destination degree so users
        # can switch to mean aggregation if they prefer.
        n_nodes = int(max(edge_index.max().item(), 0)) + 1
        deg = torch.zeros(n_nodes, dtype=torch.float32)
        deg.index_add_(0, edge_index[1].long(), torch.ones(edge_index.shape[1]))
        self.register_buffer("dst_degree", deg.clamp_min(1.0), persistent=False)

    @property
    def n_nodes(self) -> int:
        return int(self.dst_degree.shape[0])

    @property
    def n_parameters(self) -> int:
        return self.k * self.m * self.m

    def forward(self, z: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Apply one Koopman step.

        Parameters
        ----------
        z : Tensor of shape [B, N_M, m] or [N_M, m]
        normalize : bool
            If True, divide aggregated contributions by destination degree
            (mean aggregation). Default is sum aggregation, which preserves
            the spectral structure of the Koopman operator more faithfully.
        """
        squeeze = (z.dim() == 2)
        if squeeze:
            z = z.unsqueeze(0)
        b, n, m = z.shape
        assert m == self.m, f"expected latent dim {self.m}, got {m}"

        # K[edge_type] : [E, m, m]
        K_e = self.K[self.edge_type]                          # [E, m, m]
        # gather source features per edge: [B, E, m]
        z_src = z[:, self.edge_src]                           # [B, E, m]
        # contribute K_e · z_src for each edge: [B, E, m]
        contrib = torch.einsum("eij,bej->bei", K_e, z_src)
        # aggregate to destinations
        out = z.new_zeros(b, n, m)
        out.index_add_(1, self.edge_dst, contrib)
        if normalize:
            out = out / self.dst_degree.view(1, -1, 1)
        return out.squeeze(0) if squeeze else out

    def rollout(self, z0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Autoregressive Koopman rollout returning [B, T, N_M, m]."""
        z = z0
        outs = [z]
        for _ in range(n_steps):
            z = self.forward(z)
            outs.append(z)
        return torch.stack(outs, dim=1)

    # ── Spectral analysis (for NCP) ─────────────────────────────────────
    @torch.no_grad()
    def dense_matrix(self) -> torch.Tensor:
        """Materialise the (sparse) Koopman matrix as a dense [r, r] block.

        Useful only for diagnostics on small meshes — for r ≈ 1.3M it is too
        large to materialise in practice. For NCP we rely on the per-type
        sub-matrices `self.K`.
        """
        n = self.n_nodes
        r = n * self.m
        K_full = torch.zeros(r, r, device=self.K.device, dtype=self.K.dtype)
        for e in range(self.edge_src.shape[0]):
            i = int(self.edge_dst[e]); j = int(self.edge_src[e])
            t = int(self.edge_type[e])
            K_full[i * self.m:(i + 1) * self.m,
                   j * self.m:(j + 1) * self.m] += self.K[t]
        return K_full

    @torch.no_grad()
    def block_svd(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-type SVD of the K_α sub-matrices.

        Returns (U, S, V) of shapes [k, m, m], [k, m], [k, m, m] such that
        K_α = U_α diag(S_α) V_α^T. The NCP transition density is built from
        these per-type singular triples.
        """
        U, S, Vh = torch.linalg.svd(self.K, full_matrices=False)
        return U, S, Vh.transpose(-2, -1)
