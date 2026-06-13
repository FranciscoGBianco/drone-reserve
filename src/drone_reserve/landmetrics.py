"""Step 07 — landscape / habitat structure metrics.

Per-zone canopy metrics (from the CHM) and class-level fragmentation metrics
(FRAGSTATS-style, from the habitat raster), computed transparently with
scipy.ndimage — no FRAGSTATS / pylandstats dependency.

Definitions (class-level, for a focal class within a zone):
- NP   number of patches (8-connectivity)
- PD   patch density (patches per 100 ha)
- area total class area (ha) and % of zone
- MPS  mean patch size (ha)
- LPI  largest patch index (largest patch area / zone area, %)
- ED   edge density (m of class/non-class boundary per ha of zone) — counts
       borders to other in-zone classes, not the artificial footprint edge.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

from .io import resolve_path

__all__ = [
    "clip_to_geom",
    "sieve_habitat",
    "canopy_metrics",
    "class_metrics",
]


def sieve_habitat(class_arr, valid, *, min_pixels: int, nodata: int = 255,
                  connectivity: int = 8):
    """Remove patches smaller than ``min_pixels`` (minimum mapping unit), reassigning
    them to their largest neighbour — so per-pixel segmentation salt-and-pepper doesn't
    inflate patch counts. Returns an int array (``nodata`` outside ``valid``).
    """
    from rasterio.features import sieve
    arr = np.where(valid, class_arr.astype("int32"), nodata).astype("int32")
    return sieve(arr, size=int(min_pixels), connectivity=connectivity)


def clip_to_geom(raster_path, geom):
    """Clip a raster to a geometry; return (array, valid_mask, pixel_size_m, zone_area_ha).

    ``valid_mask`` is True for in-geometry, non-nodata pixels. ``zone_area_ha`` is the
    valid (analysed) area, used as the landscape denominator.
    """
    raster_path = resolve_path(raster_path)
    with rasterio.open(raster_path) as src:
        arr, transform = rio_mask(src, geom, crop=True, filled=True, nodata=src.nodata)
        nd = src.nodata
        px = abs(transform.a)
    band = arr[0].astype("float64")
    valid = np.isfinite(band)
    if nd is not None:
        valid &= band != nd
    zone_area_ha = float(valid.sum()) * px * px / 10_000
    return band, valid, px, zone_area_ha


@dataclass
class CanopyMetrics:
    zone: str
    zone_area_ha: float
    cover_2m_pct: float
    cover_3m_pct: float
    mean_canopy_height_m: float   # mean CHM where CHM >= 2 m
    p95_height_m: float
    max_height_m: float
    mean_height_all_m: float       # mean CHM over the whole zone


def canopy_metrics(chm: np.ndarray, valid: np.ndarray, zone: str,
                   zone_area_ha: float) -> CanopyMetrics:
    """Canopy cover and height metrics from a clipped CHM."""
    v = chm[valid]
    canopy2 = v >= 2.0
    canopy3 = v >= 3.0
    tall = v[v >= 2.0]
    return CanopyMetrics(
        zone=zone,
        zone_area_ha=round(zone_area_ha, 2),
        cover_2m_pct=round(100 * canopy2.mean(), 1),
        cover_3m_pct=round(100 * canopy3.mean(), 1),
        mean_canopy_height_m=round(float(tall.mean()), 2) if tall.size else float("nan"),
        p95_height_m=round(float(np.percentile(v, 95)), 2),
        max_height_m=round(float(v.max()), 2),
        mean_height_all_m=round(float(v.mean()), 2),
    )


@dataclass
class ClassMetrics:
    zone: str
    class_name: str
    area_ha: float
    pct_of_zone: float
    n_patches: int
    patch_density_per_100ha: float
    mean_patch_size_ha: float
    largest_patch_index_pct: float
    edge_density_m_per_ha: float


def class_metrics(class_arr: np.ndarray, valid: np.ndarray, *, class_id: int,
                  class_name: str, zone: str, pixel_size: float,
                  zone_area_ha: float, connectivity: int = 8) -> ClassMetrics:
    """FRAGSTATS-style class-level metrics for one class in one zone."""
    from scipy import ndimage

    px_ha = pixel_size * pixel_size / 10_000
    cls = (class_arr == class_id) & valid

    structure = np.ones((3, 3), int) if connectivity == 8 else None
    labels, n = ndimage.label(cls, structure=structure)
    if n:
        sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
        sizes_ha = sizes * px_ha
        area_ha = float(sizes_ha.sum())
        mps = float(sizes_ha.mean())
        lpi = 100 * float(sizes_ha.max()) / zone_area_ha
    else:
        area_ha = mps = lpi = 0.0

    # Edge length: class/valid-non-class 4-neighbour borders.
    nonclass = valid & ~cls
    eh = (cls[:, :-1] & nonclass[:, 1:]) | (nonclass[:, :-1] & cls[:, 1:])
    ev = (cls[:-1, :] & nonclass[1:, :]) | (nonclass[:-1, :] & cls[1:, :])
    edge_len_m = (int(eh.sum()) + int(ev.sum())) * pixel_size

    return ClassMetrics(
        zone=zone,
        class_name=class_name,
        area_ha=round(area_ha, 2),
        pct_of_zone=round(100 * area_ha / zone_area_ha, 1),
        n_patches=int(n),
        patch_density_per_100ha=round(n / zone_area_ha * 100, 1),
        mean_patch_size_ha=round(mps, 3),
        largest_patch_index_pct=round(lpi, 1),
        edge_density_m_per_ha=round(edge_len_m / zone_area_ha, 1),
    )
