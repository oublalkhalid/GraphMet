"""Per-node MLP decoder and Mesh-to-Grid stage.

The decoder mirrors the encoder: a per-node MLP maps from the latent space
Z back to mesh features, then a one-layer MP scatters those features onto
the original lat/lon grid using a mesh→grid kNN bipartite graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import _MLP, _scatter_mean


class NodeDecoder(nn.Module):
    """Per-node MLP from latent space Z back to mesh features."""

    def __init__(self, latent_dim: int, out_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        self.mlp = _MLP(latent_dim, hidden, out_dim, n_layers=n_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class Mesh2GridDecoder(nn.Module):
    """One-layer MP from mesh nodes back to the lat/lon grid.

    The bipartite edge index is the *transpose* of the grid→mesh kNN used
    by the encoder: for every grid point we have k mesh-node neighbours.
    """

    def __init__(
        self,
        mesh_in: int,
        grid_in: int,
        edge_in: int,
        hidden: int = 256,
        out_dim: int = 1,
    ):
        super().__init__()
        self.edge_mlp = _MLP(mesh_in + grid_in + edge_in, hidden, hidden)
        self.node_mlp = _MLP(grid_in + hidden, hidden, out_dim)

    def forward(
        self,
        mesh_h: torch.Tensor,           # [B, N_M, F_mesh]
        grid_x: torch.Tensor,           # [B, N_G, F_grid] (static / coords)
        edge_index: torch.Tensor,       # [2, E] rows = (mesh_idx, grid_idx)
        edge_attr: torch.Tensor,        # [E, F_edge]
    ) -> torch.Tensor:                  # [B, N_G, out_dim]
        b, n_g = grid_x.shape[0], grid_x.shape[1]
        src, dst = edge_index[0], edge_index[1]
        e = edge_attr.expand(b, -1, -1) if edge_attr.dim() == 2 else edge_attr
        mh = mesh_h[:, src]
        gx = grid_x[:, dst]
        ef = self.edge_mlp(torch.cat([mh, gx, e], dim=-1))
        agg = torch.stack(
            [_scatter_mean(ef[i], dst, n_g) for i in range(b)], dim=0
        )
        return self.node_mlp(torch.cat([grid_x, agg], dim=-1))
