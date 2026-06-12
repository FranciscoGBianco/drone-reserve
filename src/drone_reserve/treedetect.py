"""Step 06 — individual tree detection from the CHM (treetops).

Local-maxima detection with **height-scaled greedy non-maximum suppression**:
taller trees have wider crowns, so a detected apex suppresses other candidates
within an allometric crown radius ``r(h) = base + slope * h``. This is the
standard variable-window approach (after Popescu & Wynne 2004), implemented with
scipy + a KD-tree — no scikit-image needed.

Crown *delineation* (watershed → polygons, crown area) is a separate, optional
step that would add scikit-image; the treetop detection + height validation here
does not require it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from .io import resolve_path

__all__ = [
    "detect_treetops",
    "match_to_reference",
]


@dataclass
class TreetopResult:
    gdf: "gpd.GeoDataFrame"   # detected treetops: geometry (point), height (m)
    n: int
    min_height: float
    crown_base: float
    crown_slope: float


def detect_treetops(
    chm_path: str | Path,
    *,
    min_height: float = 2.0,
    smooth_sigma: float = 0.6,
    candidate_window: int = 3,
    crown_base: float = 1.0,
    crown_slope: float = 0.12,
    mask_path: str | Path | None = None,
    mask_min: float | None = None,
) -> TreetopResult:
    """Detect treetops as CHM local maxima, deduplicated by allometric crown radius.

    Steps: light Gaussian smooth (suppress speckle) -> candidate local maxima in a
    small window with CHM >= ``min_height`` -> sort by height -> greedy NMS keeping a
    peak only if no taller accepted peak lies within ``crown_base + crown_slope*h``.

    ``mask_path``/``mask_min``: optionally keep only candidates where a companion
    raster (e.g. the corrected CHM, or a forest-class raster) is >= ``mask_min`` —
    used to restrict detection to the forest parcel.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    from scipy.spatial import cKDTree
    import geopandas as gpd
    from shapely.geometry import Point

    chm_path = resolve_path(chm_path)
    with rasterio.open(chm_path) as src:
        chm = src.read(1).astype("float64")
        nd = src.nodata
        transform = src.transform
        crs = src.crs
    if nd is not None:
        chm[chm == nd] = np.nan

    valid = np.isfinite(chm)
    work = np.where(valid, chm, -1.0)
    if smooth_sigma and smooth_sigma > 0:
        work = gaussian_filter(work, sigma=smooth_sigma)

    # Candidate local maxima within a small window.
    mx = maximum_filter(work, size=candidate_window, mode="nearest")
    cand = valid & (work >= min_height) & (work == mx)

    if mask_path is not None and mask_min is not None:
        with rasterio.open(resolve_path(mask_path)) as m:
            mraw = m.read(1).astype("float64")
            mnd = m.nodata
        if mnd is not None:
            mraw[mraw == mnd] = np.nan
        cand &= np.isfinite(mraw) & (mraw >= mask_min)

    rows, cols = np.where(cand)
    heights = chm[rows, cols]
    # World coords of pixel centres.
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.asarray(xs); ys = np.asarray(ys)

    # Greedy NMS, tallest first, allometric crown radius.
    order = np.argsort(heights)[::-1]
    xy = np.column_stack([xs, ys])
    tree = cKDTree(xy)
    suppressed = np.zeros(len(order), dtype=bool)
    keep = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        r = crown_base + crown_slope * heights[idx]
        for j in tree.query_ball_point(xy[idx], r):
            if j != idx and heights[j] <= heights[idx]:
                suppressed[j] = True

    keep = np.array(keep, dtype=int)
    gdf = gpd.GeoDataFrame(
        {"height": heights[keep]},
        geometry=[Point(x, y) for x, y in xy[keep]],
        crs=crs,
    ).sort_values("height", ascending=False).reset_index(drop=True)

    return TreetopResult(gdf=gdf, n=len(gdf), min_height=min_height,
                         crown_base=crown_base, crown_slope=crown_slope)


def match_to_reference(
    detected,
    reference,
    *,
    max_dist_m: float = 2.5,
    ref_height_col: str = "RF height",
):
    """Match each reference tree to its nearest detected treetop within ``max_dist_m``.

    Returns a DataFrame (one row per reference tree) with the match distance,
    detected height, and reference height, plus a summary dict (detection rate,
    height RMSE/bias on matched trees).
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    det_xy = np.column_stack([detected.geometry.x, detected.geometry.y])
    ref_xy = np.column_stack([reference.geometry.x, reference.geometry.y])
    tree = cKDTree(det_xy)
    dist, idx = tree.query(ref_xy, k=1)

    matched = dist <= max_dist_m
    det_h = detected["height"].to_numpy()[idx]
    rows = pd.DataFrame({
        "ref_height": reference[ref_height_col].to_numpy(dtype=float),
        "match_dist_m": np.round(dist, 2),
        "matched": matched,
        "det_height": np.where(matched, np.round(det_h, 2), np.nan),
    })
    if "Species" in reference.columns:
        rows.insert(0, "Species", reference["Species"].to_numpy())

    m = matched
    resid = (det_h[m] - rows["ref_height"].to_numpy()[m]) if m.any() else np.array([])
    summary = {
        "n_reference": int(len(reference)),
        "n_detected_total": int(len(detected)),
        "detection_rate": float(m.mean()),
        "n_matched": int(m.sum()),
        "height_rmse": float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan"),
        "height_bias": float(np.mean(resid)) if resid.size else float("nan"),
    }
    return rows, summary
