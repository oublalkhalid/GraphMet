"""Icosahedral mesh construction and multi-mesh refinement.

A regular icosahedron is recursively refined `n_refine` times (by midpoint
subdivision and projection onto the unit sphere) to obtain successively finer
meshes M_0, M_1, ..., M_R.

The *multi-mesh* used by GraphMet is the union of all nodes from the finest
level M_R together with the edges of every level M_0 ... M_R. This gives the
single Koopman step access to long-range edges so information propagates
globally without depth.

Edges are typed: for each refinement level we assign one of 12 directional
classes (six oriented hex neighbours, in both directions). Self-loops are
typed by land/ocean membership, giving k = 14 interaction types in total
(matching the paper).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


# ────────────────────────────────────────────────────────────────────────────
# Icosahedron base construction
# ────────────────────────────────────────────────────────────────────────────

def _icosahedron_vertices_faces() -> Tuple[np.ndarray, np.ndarray]:
    """Return the 12 vertices and 20 faces of a unit icosahedron."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return verts, faces


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = 0.5 * (a + b)
    return m / np.linalg.norm(m)


def _refine_once(
    verts: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """One subdivision step: each triangle becomes four."""
    midpoint_cache: dict = {}
    new_verts = list(verts)

    def midpoint_idx(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in midpoint_cache:
            return midpoint_cache[key]
        m = _midpoint(verts[i], verts[j])
        idx = len(new_verts)
        new_verts.append(m)
        midpoint_cache[key] = idx
        return idx

    new_faces: List[List[int]] = []
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        ab = midpoint_idx(a, b)
        bc = midpoint_idx(b, c)
        ca = midpoint_idx(c, a)
        new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
    return np.asarray(new_verts), np.asarray(new_faces, dtype=np.int64), midpoint_cache


def _edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Return unique undirected edges as int64 [E, 2]."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    return e


# ────────────────────────────────────────────────────────────────────────────
# Mesh dataclasses
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class IcosahedralMesh:
    """A single refinement level."""

    verts: np.ndarray   # [N, 3] xyz on unit sphere
    faces: np.ndarray   # [F, 3] int64
    edges: np.ndarray   # [E, 2] int64, undirected, sorted by (min,max)
    level: int

    @property
    def n_nodes(self) -> int:
        return self.verts.shape[0]

    def latlon(self) -> np.ndarray:
        """Return (lat, lon) in degrees for every vertex."""
        x, y, z = self.verts[:, 0], self.verts[:, 1], self.verts[:, 2]
        lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))
        return np.stack([lat, lon], axis=1)


@dataclass
class MultiMesh:
    """Union of node positions at the finest level + edges from every level.

    Attributes
    ----------
    finest : IcosahedralMesh
        The finest mesh (positions used for all features).
    edge_index : torch.LongTensor [2, E_total]
        Directed edges (each undirected edge appears twice, j→i and i→j).
    edge_type : torch.LongTensor [E_total]
        Integer in [0, n_types) giving the interaction class for each edge.
    n_types : int
        Number of distinct edge-interaction types (14 in the paper).
    """

    finest: IcosahedralMesh
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    n_types: int

    @property
    def n_nodes(self) -> int:
        return self.finest.n_nodes


# ────────────────────────────────────────────────────────────────────────────
# Multi-mesh construction
# ────────────────────────────────────────────────────────────────────────────

def _edge_type_from_direction(src: np.ndarray, dst: np.ndarray) -> int:
    """Assign one of 12 directional classes to an edge on the sphere.

    Classes are defined by the great-circle bearing of the edge at its
    source node, quantised into 6 angular bins, then doubled (×2) by
    direction to produce 12 directional classes.
    """
    # Local east/north basis at src
    lat = math.asin(max(-1.0, min(1.0, float(src[2]))))
    lon = math.atan2(float(src[1]), float(src[0]))
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array(
        [-math.sin(lat) * math.cos(lon),
         -math.sin(lat) * math.sin(lon),
         math.cos(lat)]
    )
    diff = dst - src
    e_comp = float(np.dot(diff, east))
    n_comp = float(np.dot(diff, north))
    bearing = math.atan2(e_comp, n_comp)  # [-π, π]
    bin6 = int(((bearing + math.pi) / (2 * math.pi)) * 6) % 6
    # Two directions per edge are produced by caller; this assigns the bin
    return bin6


def _land_ocean_mask(latlon: np.ndarray, mask_fn=None) -> np.ndarray:
    """Return boolean array (True = land) for every vertex.

    Without an actual land–sea mask we fall back to a deterministic
    pseudo-mask based on latitude bands. Override `mask_fn(lat, lon) -> bool`
    when wiring real geography (e.g. interp from a 0.25° land mask).
    """
    if mask_fn is None:
        # Simple stand-in: poles + some longitudes flagged as land.
        lats = latlon[:, 0]
        lons = latlon[:, 1]
        return ((np.abs(lats) > 60.0) | (np.cos(np.radians(lons * 2)) > 0.0))
    return np.array([bool(mask_fn(la, lo)) for la, lo in latlon], dtype=bool)


def build_multimesh(
    n_refine: int,
    n_directional: int = 12,
    land_mask_fn=None,
) -> MultiMesh:
    """Construct the multi-mesh used by GraphMet.

    Parameters
    ----------
    n_refine : int
        Number of icosahedral refinements (paper uses 6 → 40,962 nodes).
    n_directional : int
        Number of directional edge classes (paper uses 12).
    land_mask_fn : callable or None
        Optional `(lat_deg, lon_deg) -> bool` callable for the land mask.

    Returns
    -------
    MultiMesh
    """
    verts, faces = _icosahedron_vertices_faces()
    levels: List[IcosahedralMesh] = [
        IcosahedralMesh(verts=verts, faces=faces,
                        edges=_edges_from_faces(faces), level=0)
    ]
    for lvl in range(n_refine):
        verts, faces, _ = _refine_once(verts, faces)
        levels.append(
            IcosahedralMesh(verts=verts, faces=faces,
                            edges=_edges_from_faces(faces), level=lvl + 1)
        )

    finest = levels[-1]
    n_finest = finest.n_nodes

    # Map every coarse vertex to its index in the finest mesh.
    # Refinement preserves coarse vertex indices at the start of the array,
    # so coarse indices are valid finest indices.
    src_list, dst_list, type_list = [], [], []
    for lvl, m in enumerate(levels):
        for (a, b) in m.edges:
            a, b = int(a), int(b)
            # both directions
            t_ab = _edge_type_from_direction(m.verts[a], m.verts[b])
            t_ba = _edge_type_from_direction(m.verts[b], m.verts[a])
            src_list.extend([a, b])
            dst_list.extend([b, a])
            type_list.extend([t_ab, t_ba + n_directional // 2])

    # De-duplicate (multi-mesh may produce identical edges across levels)
    eidx = np.stack([src_list, dst_list], axis=0)
    etype = np.array(type_list, dtype=np.int64)
    key = eidx[0] * (n_finest + 7) + eidx[1]
    uniq_key, uniq_idx = np.unique(key, return_index=True)
    eidx = eidx[:, uniq_idx]
    etype = etype[uniq_idx]

    # Land / ocean self-loops
    latlon = finest.latlon()
    land = _land_ocean_mask(latlon, land_mask_fn)
    self_src = np.arange(n_finest, dtype=np.int64)
    self_dst = self_src.copy()
    self_type = np.where(land, n_directional, n_directional + 1).astype(np.int64)
    eidx = np.concatenate([eidx, np.stack([self_src, self_dst], axis=0)], axis=1)
    etype = np.concatenate([etype, self_type], axis=0)

    n_types = int(etype.max()) + 1
    return MultiMesh(
        finest=finest,
        edge_index=torch.from_numpy(eidx).long(),
        edge_type=torch.from_numpy(etype).long(),
        n_types=n_types,
    )


# ────────────────────────────────────────────────────────────────────────────
# Grid ↔ mesh adjacencies (for G2M and M2G)
# ────────────────────────────────────────────────────────────────────────────

def grid2mesh_knn(
    grid_latlon: torch.Tensor,    # [N_G, 2] in degrees
    mesh_latlon: torch.Tensor,    # [N_M, 2] in degrees
    k: int = 3,
) -> torch.Tensor:
    """Build a k-nearest-neighbour bipartite edge index from grid to mesh.

    Returns a LongTensor of shape [2, k * N_G] with rows (grid_idx, mesh_idx).
    """
    g = grid_latlon.to(torch.float32)
    m = mesh_latlon.to(torch.float32)
    # Convert to xyz to avoid wraparound near the dateline
    g_xyz = _latlon_to_xyz(g)
    m_xyz = _latlon_to_xyz(m)
    # Pairwise distance, chunked along N_G to control memory
    n_g = g.shape[0]
    chunk = 4096
    out_src, out_dst = [], []
    for s in range(0, n_g, chunk):
        e = min(n_g, s + chunk)
        d = torch.cdist(g_xyz[s:e], m_xyz)              # [b, N_M]
        idx = d.topk(k, dim=1, largest=False).indices    # [b, k]
        src = torch.arange(s, e).unsqueeze(1).expand(-1, k)
        out_src.append(src.reshape(-1))
        out_dst.append(idx.reshape(-1))
    src = torch.cat(out_src).long()
    dst = torch.cat(out_dst).long()
    return torch.stack([src, dst], dim=0)


def mesh2grid_knn(
    mesh_latlon: torch.Tensor,
    grid_latlon: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """Mesh→grid kNN: for every grid point find its k nearest mesh nodes."""
    return grid2mesh_knn(grid_latlon, mesh_latlon, k=k)


def _latlon_to_xyz(latlon: torch.Tensor) -> torch.Tensor:
    lat = torch.deg2rad(latlon[:, 0])
    lon = torch.deg2rad(latlon[:, 1])
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=1)


def latitude_weights(lat_deg: torch.Tensor) -> torch.Tensor:
    """Latitude-area weights normalised to mean 1 over the input array."""
    w = torch.cos(torch.deg2rad(lat_deg))
    return w / w.mean().clamp_min(1e-12)
