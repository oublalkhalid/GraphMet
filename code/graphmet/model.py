"""GraphMet full model: Encode → Koopman step → Decode.

Pipeline (Figure 1-A of the paper):

  x^t (grid, N_G × d)
      │ G2M (one-layer MP)
      ▼
  mesh features
      │ NodeEncoder (per-node MLP)
      ▼
  z^t  (latent, N_M × m)
      │ BlockSparseKoopman (single linear step)
      ▼
  z^{t+1}
      │ NodeDecoder (per-node MLP)
      ▼
  mesh features
      │ M2G (one-layer MP)
      ▼
  x̂^{t+1} (grid, N_G × d)

Everything except the Mesh2Mesh processor mirrors GraphCast; the only
architectural novelty is the single Koopman step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .decoder import Mesh2GridDecoder, NodeDecoder
from .encoder import Grid2MeshEncoder, NodeEncoder
from .koopman import BlockSparseKoopman
from .mesh import MultiMesh, grid2mesh_knn, mesh2grid_knn


@dataclass
class GraphMetConfig:
    grid_in: int          # number of grid input channels (variables)
    grid_out: int         # number of grid output channels (often = grid_in)
    latent_dim: int = 32  # m
    mesh_feat: int = 128  # mesh feature dim before encoding to latent
    hidden: int = 256
    g2m_k: int = 3        # kNN for Grid2Mesh
    m2g_k: int = 3        # kNN for Mesh2Grid


class GraphMet(nn.Module):
    """Full GraphMet model."""

    def __init__(
        self,
        cfg: GraphMetConfig,
        mesh: MultiMesh,
        grid_latlon: torch.Tensor,    # [N_G, 2] in degrees
        static_mesh_feat: Optional[torch.Tensor] = None,   # [N_M, F_static]
    ):
        super().__init__()
        self.cfg = cfg
        self.mesh = mesh

        # Precompute bipartite edges grid↔mesh (registered as buffers)
        mesh_latlon = torch.from_numpy(mesh.finest.latlon()).float()
        self.register_buffer("grid_latlon", grid_latlon.float(), persistent=False)
        self.register_buffer("mesh_latlon", mesh_latlon, persistent=False)

        g2m = grid2mesh_knn(grid_latlon, mesh_latlon, k=cfg.g2m_k)
        m2g_grid = mesh2grid_knn(mesh_latlon, grid_latlon, k=cfg.m2g_k)
        # transpose so rows are (mesh_idx, grid_idx)
        m2g = torch.stack([m2g_grid[1], m2g_grid[0]], dim=0)
        self.register_buffer("g2m_edge_index", g2m, persistent=False)
        self.register_buffer("m2g_edge_index", m2g, persistent=False)

        # Edge features = (Δlat, Δlon, great-circle distance) — kept tiny
        self.register_buffer("g2m_edge_attr",
                             _edge_attr(grid_latlon, mesh_latlon, g2m),
                             persistent=False)
        self.register_buffer("m2g_edge_attr",
                             _edge_attr(mesh_latlon, grid_latlon, m2g),
                             persistent=False)

        # Static mesh features (always-present geographical descriptors)
        if static_mesh_feat is None:
            static_mesh_feat = torch.zeros(mesh.n_nodes, 4)
        self.register_buffer("static_mesh_feat", static_mesh_feat.float(),
                             persistent=False)

        # ── Sub-modules ────────────────────────────────────────────────
        self.g2m = Grid2MeshEncoder(
            grid_in=cfg.grid_in + 2,            # + (lat, lon) static
            edge_in=self.g2m_edge_attr.shape[-1],
            mesh_in=self.static_mesh_feat.shape[-1],
            hidden=cfg.hidden,
            out_dim=cfg.mesh_feat,
        )
        self.node_enc = NodeEncoder(cfg.mesh_feat, cfg.latent_dim,
                                    hidden=cfg.hidden, n_layers=3)
        self.koopman = BlockSparseKoopman(
            latent_dim=cfg.latent_dim,
            n_types=mesh.n_types,
            edge_index=mesh.edge_index,
            edge_type=mesh.edge_type,
        )
        self.node_dec = NodeDecoder(cfg.latent_dim, cfg.mesh_feat,
                                    hidden=cfg.hidden, n_layers=3)
        self.m2g = Mesh2GridDecoder(
            mesh_in=cfg.mesh_feat,
            grid_in=cfg.grid_in + 2,
            edge_in=self.m2g_edge_attr.shape[-1],
            hidden=cfg.hidden,
            out_dim=cfg.grid_out,
        )

    # ── Forward helpers ────────────────────────────────────────────────

    def _augment_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Append (lat, lon) channels to grid features."""
        b = x.shape[0]
        ll = self.grid_latlon.unsqueeze(0).expand(b, -1, -1) / 90.0
        return torch.cat([x, ll], dim=-1)

    def _augment_mesh(self, b: int) -> torch.Tensor:
        return self.static_mesh_feat.unsqueeze(0).expand(b, -1, -1)

    # ── Public API ────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Grid → latent  (B, N_G, d) → (B, N_M, m)."""
        b = x.shape[0]
        xg = self._augment_grid(x)
        mesh_h = self.g2m(xg, self._augment_mesh(b),
                          self.g2m_edge_index, self.g2m_edge_attr)
        return self.node_enc(mesh_h)

    def decode(self, z: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        """Latent → grid (uses x_ref only for grid coords/residual)."""
        b = z.shape[0]
        mesh_h = self.node_dec(z)
        xg = self._augment_grid(x_ref)
        return self.m2g(mesh_h, xg, self.m2g_edge_index, self.m2g_edge_attr)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        """One Koopman step in latent space."""
        return self.koopman(z)

    def forward(
        self,
        x: torch.Tensor,
        return_latents: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """One-step prediction.

        Returns
        -------
        x_hat : [B, N_G, d]
        (zt, ztp1) if `return_latents` else None
        """
        zt = self.encode(x)
        ztp1 = self.step(zt)
        x_hat = self.decode(ztp1, x)
        if return_latents:
            return x_hat, zt, ztp1
        return x_hat, None, None

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Autoregressive rollout. Returns [B, T+1, N_G, d] (including x0)."""
        z = self.encode(x0)
        outs = [x0]
        for _ in range(n_steps):
            z = self.step(z)
            outs.append(self.decode(z, x0))
        return torch.stack(outs, dim=1)


# ────────────────────────────────────────────────────────────────────────────
# Edge-attribute helper
# ────────────────────────────────────────────────────────────────────────────

def _edge_attr(src_latlon: torch.Tensor, dst_latlon: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
    """Return (Δlat/90, Δlon/180, gcd) per edge as a [E, 3] tensor."""
    s = src_latlon[edge_index[0]]
    d = dst_latlon[edge_index[1]]
    dlat = (d[:, 0] - s[:, 0]) / 90.0
    dlon = (d[:, 1] - s[:, 1]) / 180.0
    # Great-circle distance (radians, normalised by π)
    s_rad = torch.deg2rad(s); d_rad = torch.deg2rad(d)
    a = torch.sin(0.5 * (d_rad[:, 0] - s_rad[:, 0])).pow(2) + \
        torch.cos(s_rad[:, 0]) * torch.cos(d_rad[:, 0]) * \
        torch.sin(0.5 * (d_rad[:, 1] - s_rad[:, 1])).pow(2)
    gcd = 2.0 * torch.asin(torch.sqrt(a.clamp(0.0, 1.0))) / torch.pi
    return torch.stack([dlat, dlon, gcd], dim=-1).float()
