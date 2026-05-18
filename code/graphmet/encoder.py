"""Grid-to-Mesh encoder and per-node MLP encoder.

The Grid2Mesh stage aggregates features from a regular lat/lon grid onto the
finest icosahedral mesh using a small message-passing layer over the bipartite
kNN graph. A subsequent per-node MLP maps the mesh features into the latent
space Z = R^{N_M × m}.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _scatter_mean(src: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    """Mean-aggregate `src` (shape [E, F]) by destination index `idx` (shape [E]).

    Returns tensor of shape [n, F]. Falls back to a pure-PyTorch
    implementation so we do not require torch_scatter.
    """
    f = src.shape[-1]
    out = torch.zeros(n, f, device=src.device, dtype=src.dtype)
    count = torch.zeros(n, 1, device=src.device, dtype=src.dtype)
    out.index_add_(0, idx, src)
    count.index_add_(0, idx, torch.ones_like(src[:, :1]))
    return out / count.clamp_min(1.0)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class Grid2MeshEncoder(nn.Module):
    """One-layer MP that lifts grid features onto the mesh.

    Inputs
    ------
    grid_x : [B, N_G, F_grid]
    edge_index : [2, E] LongTensor, rows = (grid_idx, mesh_idx)
    edge_attr : [E, F_edge] (e.g. relative position grid→mesh)
    """

    def __init__(
        self,
        grid_in: int,
        edge_in: int,
        mesh_in: int,
        hidden: int = 256,
        out_dim: int = 128,
    ):
        super().__init__()
        self.edge_mlp = _MLP(grid_in + mesh_in + edge_in, hidden, out_dim)
        self.node_mlp = _MLP(mesh_in + out_dim, hidden, out_dim)

    def forward(
        self,
        grid_x: torch.Tensor,           # [B, N_G, F_grid]
        mesh_x: torch.Tensor,           # [B, N_M, F_mesh]
        edge_index: torch.Tensor,       # [2, E]
        edge_attr: torch.Tensor,        # [E, F_edge]
    ) -> torch.Tensor:                  # [B, N_M, out_dim]
        b, n_m = mesh_x.shape[0], mesh_x.shape[1]
        src, dst = edge_index[0], edge_index[1]
        e = edge_attr.expand(b, -1, -1) if edge_attr.dim() == 2 else edge_attr
        # gather endpoints
        gx = grid_x[:, src]                # [B, E, F_grid]
        mx = mesh_x[:, dst]                # [B, E, F_mesh]
        ef = self.edge_mlp(torch.cat([gx, mx, e], dim=-1))   # [B, E, out]
        # aggregate per mesh node
        agg = torch.stack(
            [_scatter_mean(ef[i], dst, n_m) for i in range(b)], dim=0
        )                                  # [B, N_M, out]
        return self.node_mlp(torch.cat([mesh_x, agg], dim=-1))


class NodeEncoder(nn.Module):
    """Per-node MLP from the mesh feature space into the latent space Z."""

    def __init__(self, in_dim: int, latent_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        self.mlp = _MLP(in_dim, hidden, latent_dim, n_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
