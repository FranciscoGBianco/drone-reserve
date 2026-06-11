"""Step 05 — unsupervised habitat segmentation (RGB + ExG + CHM).

No NIR is available, so no NDVI. Instead the feature stack combines colour
(RGB), an RGB pseudo-vegetation index (ExG, excess green), and the corrected
CHM — height is the strongest discriminator the poster never used for mapping
(forest canopy is tall; everything else is low).

Everything is built on the CHM's 0.5 m grid. The high-resolution ortho is read
through a ``WarpedVRT`` (streamed, never fully in memory). Clustering is
unsupervised (KMeans): it finds natural groups; a human assigns class names from
the per-cluster feature centroids (see ``cluster_centroids``).

Target classes (reliable RGB+CHM set): forest canopy, low vegetation, bare soil,
water. Trails (shape) and wetland (moisture gradient) are weak in RGB — stretch
classes handled later.
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
    "FEATURE_NAMES",
    "CLASS_NAMES",
    "FeatureStack",
    "build_feature_stack",
    "cluster_kmeans",
    "cluster_centroids",
    "assign_classes",
    "apply_class_map",
    "write_label_raster",
]

# Reliable RGB+CHM class set (Step 05). Water only assigned if a dark/flat/low
# cluster exists; trails & wetland are stretch classes handled separately.
# "bare / dry ground" (not "bare soil") because RGB+ExG cannot separate true bare
# soil from dry/senescent vegetation and leaf litter without NIR — they all read
# as low-greenness ground.
CLASS_NAMES = {0: "forest canopy", 1: "low vegetation", 2: "bare / dry ground", 3: "water"}

# Three orthogonal ecological axes — greenness, lightness, height. Using R,G,B
# directly triple-counts brightness and lets colour out-vote height; ExG +
# brightness + CHM cleanly span {forest canopy, low veg, bare soil, water}.
FEATURE_NAMES = ("ExG", "brightness", "CHM")


def _read_band_on_grid(src_path, band, ref, resampling) -> np.ndarray:
    """Read one band of ``src_path`` resampled onto ``ref``'s grid (streamed)."""
    with rasterio.open(src_path) as src:
        with WarpedVRT(src, crs=ref.crs, transform=ref.transform,
                       width=ref.width, height=ref.height, resampling=resampling) as vrt:
            arr = vrt.read(band, masked=True).astype("float64")
    return arr.filled(np.nan)


@dataclass
class FeatureStack:
    features: np.ndarray   # (n_valid, n_feat) standardised-ready raw features
    valid: np.ndarray      # (H, W) bool mask of pixels used
    shape: tuple           # (H, W)
    profile: dict          # rasterio profile of the reference grid
    names: tuple           # feature names


def build_feature_stack(
    ortho_path: str | Path,
    chm_path: str | Path,
    *,
    ref_grid_path: str | Path | None = None,
) -> FeatureStack:
    """Build the per-pixel feature matrix [R, G, B, ExG, CHM] on the CHM grid.

    ExG (excess green) uses chromatic coordinates r,g,b = R/(R+G+B) etc., so it
    is largely illumination-invariant: ExG = 2g - r - b, high for vegetation.
    Returns only valid pixels (finite RGB and CHM) plus the mask to rebuild rasters.
    """
    ortho_path = resolve_path(ortho_path)
    chm_path = resolve_path(chm_path)
    grid_path = resolve_path(ref_grid_path) if ref_grid_path is not None else chm_path

    with rasterio.open(grid_path) as ref:
        profile = ref.profile.copy()
        H, W = ref.height, ref.width
        R = _read_band_on_grid(ortho_path, 1, ref, Resampling.average)
        G = _read_band_on_grid(ortho_path, 2, ref, Resampling.average)
        B = _read_band_on_grid(ortho_path, 3, ref, Resampling.average)

    with rasterio.open(chm_path) as src:
        chm = src.read(1).astype("float64")
        nd = src.nodata
    if nd is not None:
        chm[chm == nd] = np.nan

    # Chromatic coords + ExG (guard divide-by-zero on black pixels); brightness.
    total = R + G + B
    with np.errstate(invalid="ignore", divide="ignore"):
        r = R / total
        g = G / total
        b = B / total
    exg = 2 * g - r - b
    brightness = total / 3.0

    valid = (
        np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
        & np.isfinite(chm) & np.isfinite(exg) & (total > 0)
    )
    feats = np.column_stack([exg[valid], brightness[valid], chm[valid]])
    return FeatureStack(features=feats, valid=valid, shape=(H, W),
                        profile=profile, names=FEATURE_NAMES)


def cluster_kmeans(stack: FeatureStack, k: int = 6, *, seed: int = 0) -> np.ndarray:
    """Standardise features and run KMeans. Returns a (H, W) int label raster
    (cluster id >= 0 on valid pixels, -1 elsewhere)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(stack.features)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cl = km.fit_predict(X)

    labels = np.full(stack.shape, -1, dtype=np.int16)
    labels[stack.valid] = cl.astype(np.int16)
    return labels


def cluster_centroids(stack: FeatureStack, labels: np.ndarray) -> "pd.DataFrame":
    """Per-cluster mean of each (raw, un-standardised) feature + pixel count.

    This is what a human reads to assign class names: high CHM -> forest canopy;
    low CHM + high ExG -> low vegetation; low CHM + low ExG + bright -> bare soil;
    low CHM + dark -> water.
    """
    import pandas as pd

    cl = labels[stack.valid]
    df = pd.DataFrame(stack.features, columns=list(stack.names))
    df["cluster"] = cl
    summary = df.groupby("cluster").mean()
    summary["n_pixels"] = df.groupby("cluster").size()
    px_area = abs(stack.profile["transform"].a) * abs(stack.profile["transform"].e)
    summary["area_ha"] = summary["n_pixels"] * px_area / 10_000
    return summary.sort_values("CHM", ascending=False)


def assign_classes(
    centroids,
    *,
    chm_forest: float = 3.0,
    exg_veg: float = 0.05,
    water_brightness: float | None = None,
) -> dict:
    """Map each cluster id -> class id (see ``CLASS_NAMES``) by its centroid.

    Transparent rules: CHM >= ``chm_forest`` -> forest canopy; else a dark, low,
    non-green cluster (brightness < ``water_brightness``, only if that arg is set)
    -> water; else ExG >= ``exg_veg`` -> low vegetation; else -> bare soil.
    """
    mapping = {}
    for cid, row in centroids.iterrows():
        if row["CHM"] >= chm_forest:
            mapping[int(cid)] = 0
        elif (water_brightness is not None and row["ExG"] < exg_veg
              and row["brightness"] < water_brightness):
            mapping[int(cid)] = 3
        elif row["ExG"] >= exg_veg:
            mapping[int(cid)] = 1
        else:
            mapping[int(cid)] = 2
    return mapping


def apply_class_map(labels: np.ndarray, mapping: dict) -> np.ndarray:
    """Remap a cluster-label raster (-1 nodata) to class ids using ``mapping``."""
    out = np.full(labels.shape, -1, dtype=np.int16)
    for cid, cls in mapping.items():
        out[labels == cid] = cls
    return out


def write_label_raster(labels: np.ndarray, profile: dict, out_path: str | Path,
                       *, nodata: int = 255) -> str:
    """Write an integer class/cluster raster (uint8). Values < 0 become nodata."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.where(labels < 0, nodata, labels).astype("uint8")
    prof = profile.copy()
    prof.update(dtype="uint8", count=1, nodata=nodata, compress="deflate")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr, 1)
    return str(out_path)
