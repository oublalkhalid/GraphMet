"""CERRA regional dataset wrapper (5.5 km, European domain).

Same interface as ERA5Pairs but restricted to the CERRA Lambert projection
window:  30°N – 75°N, 35°W – 45°E.

When real CERRA data is not present we generate a smaller synthetic field
so the regional fine-tuning script can be smoke-tested end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import xarray as xr
    _HAS_XR = True
except Exception:        # pragma: no cover
    _HAS_XR = False


DEFAULT_CERRA_VARIABLES: Tuple[str, ...] = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
)


class CERRAPairs(Dataset):
    LAT_MIN, LAT_MAX = 30.0, 75.0
    LON_MIN, LON_MAX = -35.0, 45.0

    def __init__(
        self,
        root: str | Path,
        variables: Sequence[str] = DEFAULT_CERRA_VARIABLES,
        lead_hours: int = 6,
        years: Optional[Sequence[int]] = None,
    ):
        self.root = str(root)
        self.variables = list(variables)
        self.lead_hours = int(lead_hours)
        self.years = list(years) if years else None

        if self.root == "synthetic" or not _HAS_XR:
            self._mode = "synthetic"
            self._setup_synthetic()
        else:                                              # pragma: no cover
            self._mode = "xarray"
            self._setup_xarray()

    def _setup_synthetic(self) -> None:
        self.n_samples = 512
        self.n_lat = 80           # ~0.55° step over 45° lat span
        self.n_lon = 144          # ~0.55° step over 80° lon span
        rng = np.random.default_rng(1)
        base = rng.standard_normal((self.n_samples + 1, self.n_lat, self.n_lon,
                                    len(self.variables))).astype(np.float32)
        self._states = np.cumsum(base * 0.05, axis=0)
        lat = np.linspace(self.LAT_MAX, self.LAT_MIN, self.n_lat)
        lon = np.linspace(self.LON_MIN, self.LON_MAX, self.n_lon)
        lon_g, lat_g = np.meshgrid(lon, lat)
        self.latlon = np.stack([lat_g.reshape(-1), lon_g.reshape(-1)], axis=1)

    def _setup_xarray(self) -> None:                       # pragma: no cover
        ds = xr.open_zarr(self.root) if Path(self.root).suffix in {"", ".zarr"} \
             else xr.open_mfdataset(f"{self.root}/*.nc", combine="by_coords")
        ds = ds.sel(latitude=slice(self.LAT_MAX, self.LAT_MIN),
                    longitude=slice(self.LON_MIN, self.LON_MAX))
        if self.years is not None:
            ds = ds.sel(time=ds["time.year"].isin(self.years))
        self._ds = ds
        self.n_samples = ds.sizes["time"] - 1
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        lon_g, lat_g = np.meshgrid(lon, lat)
        self.latlon = np.stack([lat_g.reshape(-1), lon_g.reshape(-1)], axis=1)

    def __len__(self) -> int:
        return self.n_samples - 1

    def _read_grid(self, t_idx: int) -> np.ndarray:
        if self._mode == "synthetic":
            return self._states[t_idx]
        arrs = [self._ds[v].isel(time=t_idx).values        # pragma: no cover
                for v in self.variables]
        return np.stack(arrs, axis=-1).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        step = max(1, self.lead_hours // 6)
        xt = self._read_grid(idx)
        xtp = self._read_grid(idx + step)
        xt = torch.from_numpy(xt.reshape(-1, xt.shape[-1])).float()
        xtp = torch.from_numpy(xtp.reshape(-1, xtp.shape[-1])).float()
        return xt, xtp

    def grid_latlon(self) -> torch.Tensor:
        return torch.from_numpy(self.latlon).float()

    def latitude_weights(self) -> torch.Tensor:
        lats = torch.from_numpy(self.latlon[:, 0]).float()
        w = torch.cos(torch.deg2rad(lats))
        return w / w.mean().clamp_min(1e-6)
