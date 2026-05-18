"""ERA5 dataset wrapper for one-step (x^t, x^{t+1}) pairs.

The dataset expects ERA5 reanalysis arranged as netCDF or zarr stores
following the WeatherBench-2 convention (https://weatherbench2.readthedocs.io).

We assume a 0.25° regular lat/lon grid (721 × 1440 = 1,038,240 grid cells)
sampled every 6 hours. Each sample returns the full state at time t and at
t + Δt, plus the latitude weights for the loss.

To keep this file standalone we use `xarray` only optionally — if xarray
is missing we fall back to a synthetic NumPy generator so the rest of the
pipeline can still be smoke-tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import xarray as xr
    _HAS_XR = True
except Exception:        # pragma: no cover
    _HAS_XR = False


DEFAULT_VARIABLES: Tuple[str, ...] = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_500",
    "temperature_850",
    "specific_humidity_700",
)


class ERA5Pairs(Dataset):
    """One-step ERA5 pairs (x^t, x^{t+1}).

    Parameters
    ----------
    root : str
        Path to a directory containing ERA5 zarr/netCDF, or "synthetic"
        to generate fake data of the right shape (useful for unit tests).
    variables : sequence of str
        Variable names to include. Default = surface + a few upper-air.
    lead_hours : int
        Forecast step Δt in hours (paper uses 6 h).
    years : sequence of int or None
        Restrict to these years. If None, use every year present.
    """

    def __init__(
        self,
        root: str | Path,
        variables: Sequence[str] = DEFAULT_VARIABLES,
        lead_hours: int = 6,
        years: Optional[Sequence[int]] = None,
        normalize: bool = True,
    ):
        self.root = str(root)
        self.variables = list(variables)
        self.lead_hours = int(lead_hours)
        self.years = list(years) if years else None
        self.normalize = normalize

        if self.root == "synthetic" or not _HAS_XR:
            self._mode = "synthetic"
            self._setup_synthetic()
        else:
            self._mode = "xarray"
            self._setup_xarray()

    # ── Synthetic mode (smoke tests) ─────────────────────────────────

    def _setup_synthetic(self) -> None:
        self.n_samples = 2048
        self.n_lat = 32
        self.n_lon = 64
        rng = np.random.default_rng(0)
        # Smooth synthetic field with a slow drift to mimic Markov dynamics
        base = rng.standard_normal((self.n_samples + 1, self.n_lat, self.n_lon,
                                    len(self.variables))).astype(np.float32)
        self._states = np.cumsum(base * 0.05, axis=0)
        lat = np.linspace(90 - 90 / self.n_lat, -90 + 90 / self.n_lat, self.n_lat)
        lon = np.linspace(0, 360 - 360 / self.n_lon, self.n_lon)
        lon_g, lat_g = np.meshgrid(lon, lat)
        self.latlon = np.stack([lat_g.reshape(-1), lon_g.reshape(-1)], axis=1)

    # ── Real-data mode (xarray) ──────────────────────────────────────

    def _setup_xarray(self) -> None:                          # pragma: no cover
        ds = xr.open_zarr(self.root) if Path(self.root).suffix in {"", ".zarr"} \
             else xr.open_mfdataset(f"{self.root}/*.nc", combine="by_coords")
        if self.years is not None:
            ds = ds.sel(time=ds["time.year"].isin(self.years))
        self._ds = ds
        self.n_samples = ds.sizes["time"] - (self.lead_hours // 6)
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        self.n_lat, self.n_lon = lat.size, lon.size
        lon_g, lat_g = np.meshgrid(lon, lat)
        self.latlon = np.stack([lat_g.reshape(-1), lon_g.reshape(-1)], axis=1)
        if self.normalize:
            self._mean = {v: float(ds[v].mean().values) for v in self.variables}
            self._std = {v: float(ds[v].std().values) for v in self.variables}

    # ── Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_samples - 1

    def _read_grid(self, t_idx: int) -> np.ndarray:
        if self._mode == "synthetic":
            return self._states[t_idx]                     # [lat, lon, d]
        arrs = []
        for v in self.variables:                            # pragma: no cover
            a = self._ds[v].isel(time=t_idx).values
            if self.normalize:
                a = (a - self._mean[v]) / max(self._std[v], 1e-6)
            arrs.append(a)
        return np.stack(arrs, axis=-1).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        step = max(1, self.lead_hours // 6)
        xt = self._read_grid(idx)
        xtp = self._read_grid(idx + step)
        xt = torch.from_numpy(xt.reshape(-1, xt.shape[-1])).float()
        xtp = torch.from_numpy(xtp.reshape(-1, xtp.shape[-1])).float()
        return xt, xtp

    # ── Helpers exposed to the model ────────────────────────────────

    def grid_latlon(self) -> torch.Tensor:
        return torch.from_numpy(self.latlon).float()

    def latitude_weights(self) -> torch.Tensor:
        lats = torch.from_numpy(self.latlon[:, 0]).float()
        w = torch.cos(torch.deg2rad(lats))
        return w / w.mean().clamp_min(1e-6)
