"""Step 04 — Canopy Height Model (CHM = DSM - DTM).

The DSM (Pix4D, ~3.4 cm) and the DTMs (0.5 m and others) live on different grids,
so we resample everything onto one common grid and difference. Resampling uses
``WarpedVRT`` so the large DSM is read block-wise through the warp — it is never
loaded into memory at full resolution (project rule: tile, don't load).

Per DTM variant we get a CHM, which lets us both reproduce the poster's
VANT-CHM / dGNSS-CHM comparison and add the bias-corrected hybrid CHM.

Validation samples each tree's height as the **maximum CHM within a crown radius**
of its dGNSS position — the GPS point marks the trunk, while the canopy apex sits
a little off to the side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from .io import resolve_path

__all__ = [
    "compute_chm",
    "sample_chm_heights",
]


def _read_on_grid(src_path, ref, resampling: Resampling) -> np.ndarray:
    """Read ``src_path`` resampled onto the grid of an open ``ref`` dataset.

    Uses WarpedVRT, so only the (small) output grid is materialised; the source
    is streamed block-wise. Returns a float64 array with nodata as NaN.
    """
    with rasterio.open(src_path) as src:
        with WarpedVRT(
            src,
            crs=ref.crs,
            transform=ref.transform,
            width=ref.width,
            height=ref.height,
            resampling=resampling,
        ) as vrt:
            arr = vrt.read(1, masked=True).astype("float64")
    return arr.filled(np.nan)


@dataclass
class CHMResult:
    out_path: str
    grid_from: str
    n_valid: int
    chm_min: float
    chm_max: float
    chm_mean: float


def compute_chm(
    dsm_path: str | Path,
    dtm_path: str | Path,
    out_path: str | Path,
    *,
    ref_grid_path: str | Path | None = None,
    dsm_resampling: Resampling = Resampling.bilinear,
    dtm_resampling: Resampling = Resampling.bilinear,
    clamp_negative: bool = True,
) -> CHMResult:
    """Compute ``CHM = DSM - DTM`` on a common grid and write it.

    The grid is taken from ``ref_grid_path`` if given, else from ``dtm_path`` —
    so by default the CHM lands on the (gap-free, 0.5 m) DTM grid. DSM and DTM are
    each warped onto that grid; the DSM warp is streamed (never fully in memory).

    ``clamp_negative`` sets physically-impossible negative heights to 0 while
    preserving nodata (NaN) — small negatives are DSM/DTM noise on bare ground.
    """
    dsm_path = resolve_path(dsm_path)
    dtm_path = resolve_path(dtm_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path = resolve_path(ref_grid_path) if ref_grid_path is not None else dtm_path

    with rasterio.open(grid_path) as ref:
        profile = ref.profile.copy()
        dsm = _read_on_grid(dsm_path, ref, dsm_resampling)
        dtm = _read_on_grid(dtm_path, ref, dtm_resampling)

    chm = dsm - dtm  # NaN where either input is nodata
    if clamp_negative:
        # set negatives to 0, keep NaN (NaN < 0 is False, so NaN is preserved)
        chm = np.where(chm < 0, 0.0, chm)

    nodata = -9999.0
    out = np.where(np.isfinite(chm), chm, nodata).astype("float32")

    profile.update(dtype="float32", count=1, nodata=nodata,
                   compress="deflate", predictor=3)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)

    valid = chm[np.isfinite(chm)]
    return CHMResult(
        out_path=str(out_path),
        grid_from=str(grid_path.name),
        n_valid=int(valid.size),
        chm_min=float(valid.min()) if valid.size else float("nan"),
        chm_max=float(valid.max()) if valid.size else float("nan"),
        chm_mean=float(valid.mean()) if valid.size else float("nan"),
    )


def sample_chm_heights(
    chm_path: str | Path,
    gdf_points,
    *,
    radius_m: float = 1.5,
    stat: str = "max",
) -> np.ndarray:
    """Sample a CHM at each point as the ``stat`` over a disc of ``radius_m``.

    ``stat='max'`` (default) gives the crown apex near the trunk position — the
    standard way to read a tree's height off a CHM. Returns NaN where no valid
    CHM pixels fall in the window.
    """
    chm_path = resolve_path(chm_path)
    with rasterio.open(chm_path) as src:
        arr = src.read(1).astype("float64")
        nd = src.nodata
        transform = src.transform
        inv = ~transform
        px = abs(transform.a)
        H, W = arr.shape

    if nd is not None:
        arr[arr == nd] = np.nan
    rad = max(1, int(round(radius_m / px)))

    reducer = {"max": np.nanmax, "mean": np.nanmean, "median": np.nanmedian,
               "p95": lambda a: np.nanpercentile(a, 95)}[stat]

    out = np.full(len(gdf_points), np.nan, dtype=float)
    for i, geom in enumerate(gdf_points.geometry):
        c, r = inv * (geom.x, geom.y)
        c, r = int(round(c)), int(round(r))
        r0, r1 = max(0, r - rad), min(H, r + rad + 1)
        c0, c1 = max(0, c - rad), min(W, c + rad + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        win = arr[r0:r1, c0:c1]
        if np.isfinite(win).any():
            out[i] = float(reducer(win))
    return out
