"""DTM construction and validation for the drone-reserve project.

Step 02 of the pipeline:

1. Build photogrammetric DTMs from the Pix4D dense point clouds using
   ground filters (CSF and SMRF), via PDAL pipelines.
2. Sample any DTM raster at the 42 dGNSS validation points.
3. Report residual statistics (MAE, RMSE, ME, R²) overall and stratified.
4. Derive a continuous canopy-density score at each dGNSS point from the
   point cloud, so the Alta/Baja split from the 2025 poster can be both
   reproduced (binary) and extended (continuous).

Per-zone vertical/horizontal shifts are applied **inside** the PDAL
pipeline via ``filters.transformation`` — no parallel shifted-LAS copies
on disk. Constants come from the calibration CSVs at ``data/talar.csv`` /
``data/pastizal.csv``; see ``project_vertical_reference.md`` in memory.

PDAL is imported lazily inside :func:`run_pipeline` so this module stays
importable from the pip ``.venv`` (which doesn't carry PDAL). Validation
and density helpers work in both environments.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio

from .io import resolve_path

__all__ = [
    "ZONES",
    "las_paths_for_zone",
    "build_csf_pipeline",
    "build_smrf_pipeline",
    "run_pipeline",
    "sample_raster_at_points",
    "residual_stats",
    "canopy_density_at_points",
    "fill_dtm_voids",
]


# ---------------------------------------------------------------------------
# Per-zone calibration
# ---------------------------------------------------------------------------

# (dx, dy, dz) in metres — Pix4D local → real elevation (WGS84 / UTM 21S).
# Source: median of GNSS−Drone columns in data/{talar,pastizal}.csv;
# verified against the median Z delta between Pix4D-original rasters and
# the vertically-registered root copies in step 01.
ZONES: dict[str, dict] = {
    "talar": {
        "shift": (1.11, 1.80, 126.60),
        "las_glob": "data/drone/talar_120m/products/"
                    "Talar_120m_20250521_group1_densified_point_cloud_part_*.las",
        "area_id": 1,  # Polygon id in data/Area.shp
    },
    "pastizal": {
        "shift": (-2.37, 0.11, 118.17),
        "las_glob": "data/drone/pastizal_120m/products/"
                    "Pastizal_120m_20250521_group1_densified_point_cloud_part_*.las",
        "area_id": 2,
    },
}


def las_paths_for_zone(zone: str, repo_root: str | Path = ".") -> list[str]:
    """Resolve the LAS parts for ``zone`` relative to ``repo_root``."""
    if zone not in ZONES:
        raise KeyError(f"Unknown zone {zone!r}; expected one of {list(ZONES)}")
    pattern = str(Path(repo_root) / ZONES[zone]["las_glob"])
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No LAS files match {pattern}")
    return paths


def _shift_matrix(dx: float, dy: float, dz: float) -> str:
    """Row-major 4×4 translation matrix as a space-separated string (PDAL format)."""
    return f"1 0 0 {dx}  0 1 0 {dy}  0 0 1 {dz}  0 0 0 1"


# ---------------------------------------------------------------------------
# PDAL pipelines
# ---------------------------------------------------------------------------


def _gdal_writer(out_tif: str | Path, resolution: float, output_type: str = "idw") -> dict:
    """A writers.gdal stage for DTM-from-ground-returns.

    Kept deliberately *strict* (PDAL's default IDW radius ~= resolution*sqrt(2)).
    A larger radius would fill gaps but smooths the well-measured cells and
    degrades accuracy at the dGNSS points (tested: radius 3 m took talar coverage
    63%->98% but RMSE 0.43->0.61). Instead we keep accurate measured cells here
    and fill the empty cells separately with ``fill_dtm_voids`` (which leaves
    measured cells untouched), getting continuity without sacrificing fidelity.
    """
    return {
        "type": "writers.gdal",
        "filename": str(out_tif),
        "resolution": float(resolution),
        "output_type": output_type,
        "gdaldriver": "GTiff",
        "gdalopts": "TILED=YES,COMPRESS=DEFLATE,PREDICTOR=3",
        "data_type": "float32",
        "nodata": -9999.0,
    }


def build_csf_pipeline(
    zone: str,
    out_tif: str | Path,
    *,
    repo_root: str | Path = ".",
    resolution: float = 0.5,
    cloth_resolution: float = 0.5,
    rigidness: int = 2,
    smooth: bool = True,
    hdiff: float = 0.5,
    iterations: int = 500,
    threshold: float = 0.5,
    step: float = 0.65,
) -> dict:
    """Build a PDAL pipeline dict: LAS → shift → CSF → keep ground class → IDW raster.

    Parameter names match PDAL's ``filters.csf`` (Zhang et al. 2016 wrapped by PDAL).
    Notes:
    - ``cloth_resolution`` maps to PDAL's ``resolution`` (cloth sampling, m).
    - ``rigidness``: 1=very rigid, 2=rigid, 3=less rigid. Talar (dense canopy) often
      benefits from rigidness=2 with a fine cloth.
    - ``threshold`` is the classification distance threshold (m).
    - ``step`` is the simulation time step.
    """
    dx, dy, dz = ZONES[zone]["shift"]
    las = las_paths_for_zone(zone, repo_root)
    return {
        "pipeline": [
            *[{"type": "readers.las", "filename": p} for p in las],
            {"type": "filters.merge"},
            {"type": "filters.transformation", "matrix": _shift_matrix(dx, dy, dz)},
            {
                "type": "filters.csf",
                "resolution": cloth_resolution,
                "rigidness": rigidness,
                "smooth": smooth,
                "hdiff": hdiff,
                "iterations": iterations,
                "threshold": threshold,
                "step": step,
            },
            {"type": "filters.range", "limits": "Classification[2:2]"},
            _gdal_writer(out_tif, resolution),
        ]
    }


def build_smrf_pipeline(
    zone: str,
    out_tif: str | Path,
    *,
    repo_root: str | Path = ".",
    resolution: float = 0.5,
    scalar: float = 1.2,
    slope: float = 0.15,
    threshold: float = 0.45,
    window: float = 16.0,
) -> dict:
    """Build a PDAL pipeline dict: LAS → shift → SMRF → keep ground class → IDW raster.

    Defaults are PDAL's stock values (Pingel et al. 2013). Adjust ``window``
    (largest object size in m) to match the densest expected canopy gap.
    """
    dx, dy, dz = ZONES[zone]["shift"]
    las = las_paths_for_zone(zone, repo_root)
    return {
        "pipeline": [
            *[{"type": "readers.las", "filename": p} for p in las],
            {"type": "filters.merge"},
            {"type": "filters.transformation", "matrix": _shift_matrix(dx, dy, dz)},
            {
                "type": "filters.smrf",
                "scalar": scalar,
                "slope": slope,
                "threshold": threshold,
                "window": window,
            },
            {"type": "filters.range", "limits": "Classification[2:2]"},
            _gdal_writer(out_tif, resolution),
        ]
    }


def run_pipeline(pipeline_dict: dict, *, verbose: bool = True) -> "pdal.Pipeline":
    """Execute a PDAL pipeline. Lazy-imports ``pdal``.

    Raises ImportError with an actionable message if PDAL isn't installed
    (typical when running from the pip ``.venv`` — use the conda env).
    """
    try:
        import pdal
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PDAL is not installed in this environment. Step 02 requires the "
            "conda env — see README ('Option B') and run "
            "`conda env create -f environment.yml` then activate it."
        ) from e

    p = pdal.Pipeline(json.dumps(pipeline_dict))
    if verbose:
        stages = [s["type"] for s in pipeline_dict["pipeline"]]
        print(f"  PDAL stages: {' → '.join(stages)}")
    count = p.execute()
    if verbose:
        print(f"  Points processed: {count:,}")
    return p


# ---------------------------------------------------------------------------
# Raster sampling + residual statistics
# ---------------------------------------------------------------------------


def sample_raster_at_points(raster_path: str | Path, gdf) -> np.ndarray:
    """Sample a single-band raster at each point geometry.

    Returns a 1-D float64 array of length ``len(gdf)``. Pixels that read
    back as nodata become ``np.nan``. The point CRS must match the raster
    CRS — we check it and raise on mismatch (no silent reprojection).
    """
    with rasterio.open(raster_path) as src:
        rcrs = src.crs
        if gdf.crs is not None and rcrs is not None and gdf.crs != rcrs:
            raise ValueError(
                f"CRS mismatch: points={gdf.crs}  raster={rcrs}. "
                "Reproject the points first; we don't silently transform."
            )
        nd = src.nodata
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        out = np.empty(len(coords), dtype=np.float64)
        for i, (val,) in enumerate(src.sample(coords)):
            if nd is not None and val == nd:
                out[i] = np.nan
            elif not np.isfinite(val):
                out[i] = np.nan
            else:
                out[i] = float(val)
    return out


@dataclass
class ResidualStats:
    n: int
    mae: float
    rmse: float
    bias: float   # mean(pred - ref); + means pred > ref
    r2: float

    def as_dict(self) -> dict:
        return {"n": self.n, "MAE": self.mae, "RMSE": self.rmse,
                "ME": self.bias, "R2": self.r2}


def residual_stats(pred_z: np.ndarray, ref_z: np.ndarray) -> ResidualStats:
    """Compute MAE, RMSE, mean-error (bias) and R² between predicted and reference Z.

    R² here is the coefficient of determination 1 − SS_res / SS_tot,
    treating ``ref_z`` as the ground truth. This matches the poster's
    convention for the DTM-comparison table.
    """
    pred = np.asarray(pred_z, dtype=float)
    ref = np.asarray(ref_z, dtype=float)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, ref={ref.shape}")

    mask = np.isfinite(pred) & np.isfinite(ref)
    n = int(mask.sum())
    if n == 0:
        return ResidualStats(0, float("nan"), float("nan"), float("nan"), float("nan"))

    r = pred[mask] - ref[mask]
    y = ref[mask]
    ss_res = float(np.sum(r ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return ResidualStats(
        n=n,
        mae=float(np.mean(np.abs(r))),
        rmse=float(np.sqrt(np.mean(r ** 2))),
        bias=float(np.mean(r)),
        r2=float(r2),
    )


# ---------------------------------------------------------------------------
# Canopy density at points (LAS-derived)
# ---------------------------------------------------------------------------


def canopy_density_at_points(
    zone: str,
    gdf_points,
    *,
    repo_root: str | Path = ".",
    radius_m: float = 3.0,
    height_above_ground_m: float = 2.0,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """For each point in ``gdf_points``, count LAS returns within ``radius_m``
    (XY distance) whose Z exceeds (point elevation + ``height_above_ground_m``).

    Reads the LAS files for ``zone`` in chunks, applies the per-zone shift on
    the fly, and keeps only points inside the dGNSS bbox+buffer to bound
    memory. Builds a single 2-D KDTree at the end.

    The point GeoDataFrame must carry an ``Elevation`` column with the
    dGNSS-measured ground elevation in metres (same datum as the shifted LAS).
    """
    import laspy
    from scipy.spatial import cKDTree

    if "Elevation" not in gdf_points.columns:
        raise KeyError("gdf_points must have an 'Elevation' column (dGNSS ground Z).")

    dx, dy, dz = ZONES[zone]["shift"]
    las_files = las_paths_for_zone(zone, repo_root)

    minx, miny, maxx, maxy = gdf_points.total_bounds
    minx -= radius_m; maxx += radius_m
    miny -= radius_m; maxy += radius_m

    xs, ys, zs = [], [], []
    n_total = 0
    for fp in las_files:
        with laspy.open(fp) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                X = np.asarray(chunk.x) + dx
                Y = np.asarray(chunk.y) + dy
                Z = np.asarray(chunk.z) + dz
                in_bbox = (X >= minx) & (X <= maxx) & (Y >= miny) & (Y <= maxy)
                if in_bbox.any():
                    xs.append(X[in_bbox])
                    ys.append(Y[in_bbox])
                    zs.append(Z[in_bbox])
                n_total += len(chunk)

    if not xs:
        print(f"  No LAS returns fell inside the dGNSS bbox for zone={zone}")
        return np.zeros(len(gdf_points), dtype=int)

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    Z = np.concatenate(zs)
    print(f"  Scanned {n_total:,} LAS points; kept {len(X):,} inside bbox+buffer "
          f"({100 * len(X) / max(n_total, 1):.1f}%).")

    tree = cKDTree(np.column_stack([X, Y]))
    qxy = np.column_stack([
        [g.x for g in gdf_points.geometry],
        [g.y for g in gdf_points.geometry],
    ])
    ref_z = gdf_points["Elevation"].to_numpy(dtype=float)

    idx_lists = tree.query_ball_point(qxy, r=radius_m)
    densities = np.empty(len(gdf_points), dtype=int)
    for i, idx in enumerate(idx_lists):
        if not idx:
            densities[i] = 0
            continue
        densities[i] = int((Z[idx] > (ref_z[i] + height_above_ground_m)).sum())

    return densities


# ---------------------------------------------------------------------------
# Void filling (continuous DTM for the CHM)
# ---------------------------------------------------------------------------


@dataclass
class FillResult:
    out_path: str
    confidence_path: str | None
    pct_measured: float       # share of footprint that was measured ground
    pct_filled: float         # share interpolated by the fill
    pct_unfilled: float       # share still nodata (beyond search distance)
    max_fill_distance_m: float


def fill_dtm_voids(
    src_path: str | Path,
    out_path: str | Path,
    *,
    max_search_distance_px: int = 100,
    smoothing_iterations: int = 0,
    clip_geometry=None,
    write_confidence: bool = True,
) -> FillResult:
    """Fill nodata gaps in a DTM by interpolation, leaving measured cells intact.

    Why this and not a larger IDW radius at rasterization time: a wider IDW radius
    fills gaps but *smooths the measured cells too*, degrading accuracy at the
    dGNSS points (tested: radius 3 m took RMSE 0.43 -> 0.61). ``rasterio.fill.fillnodata``
    only touches empty cells, so measured ground keeps its fidelity (RMSE 0.43 -> 0.44)
    while coverage reaches ~100%.

    Honesty: a companion **confidence raster** is written giving each pixel's distance
    (m) to the nearest *measured* ground cell — 0 where measured, growing into filled
    regions — so the report can flag interpolated terrain (important under the dense
    canopy where ground was never seen).

    Parameters
    ----------
    max_search_distance_px
        Max pixels ``fillnodata`` searches outward. Cells farther than this from any
        measured ground stay nodata (we don't invent terrain across huge voids).
    clip_geometry
        Optional iterable of GeoJSON-like geoms (e.g. the flight footprint). Output is
        masked to it so the DTM doesn't bleed past the flown area.
    """
    from rasterio.fill import fillnodata
    from rasterio.features import geometry_mask
    from scipy.ndimage import distance_transform_edt

    src_path = resolve_path(src_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        arr = src.read(1).astype("float32")
        nd = src.nodata if src.nodata is not None else -9999.0
        profile = src.profile.copy()
        transform = src.transform
        px = abs(transform.a)
        shape = arr.shape

    measured = np.isfinite(arr) & (arr != nd)

    # Optional footprint clip — anything outside is not part of the product.
    if clip_geometry is not None:
        inside = geometry_mask(clip_geometry, out_shape=shape, transform=transform, invert=True)
    else:
        inside = np.ones(shape, dtype=bool)

    filled = fillnodata(
        arr.copy(),
        mask=measured.astype(np.uint8),
        max_search_distance=float(max_search_distance_px),
        smoothing_iterations=smoothing_iterations,
    )
    # Restrict to footprint; mark anything still empty as nodata.
    still_empty = ~np.isfinite(filled) | (filled == nd)
    out = np.where(inside & ~still_empty, filled, nd).astype("float32")

    profile.update(dtype="float32", count=1, nodata=nd, compress="deflate", predictor=3)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)

    # Distance-to-measured confidence layer (m); nodata outside footprint / unfilled.
    conf_path = None
    dist_m = distance_transform_edt(~measured) * px
    if write_confidence:
        conf = np.where((out != nd), dist_m, nd).astype("float32")
        conf_path = str(Path(out_path).with_name(Path(out_path).stem + "_confidence.tif"))
        cprof = profile.copy()
        with rasterio.open(conf_path, "w", **cprof) as dst:
            dst.write(conf, 1)

    foot = int(inside.sum())
    n_meas = int((measured & inside).sum())
    n_out = int((out != nd).sum())
    n_filled = n_out - n_meas
    return FillResult(
        out_path=str(out_path),
        confidence_path=conf_path,
        pct_measured=100 * n_meas / foot,
        pct_filled=100 * n_filled / foot,
        pct_unfilled=100 * (foot - n_out) / foot,
        max_fill_distance_m=float(dist_m[(out != nd) & ~measured].max()) if n_filled else 0.0,
    )
