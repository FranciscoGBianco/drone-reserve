"""Step 03 — dGNSS bias-correction of a photogrammetric DTM.

The poster recommended *substituting* dGNSS-derived ground in dense canopy.
Step 02 showed that's unnecessary: SMRF (and a retuned CSF) already recover the
ground to ~0.3-0.4 m. So instead of splicing surfaces, we use the 42 dGNSS
points as **control** to model and remove the base DTM's residual:

    corrected(x, y) = base(x, y) + C(x, y)

where ``C`` is a correction surface fitted to the residuals (dGNSS_z − base_z)
at the control points.

Two correction families are supported:

- **Polynomial trend** (``CorrectionModel("poly", degree=d)``). degree 0 = pure
  debias; degree 1 = affine (constant + tilt); degree 2 = quadratic. Empirically
  the residual here is dominated by a degree-1 **datum tilt**, so the affine model
  generalizes best under leave-one-out. A trend is *global* — applied across the
  whole raster (tapering a global trend would create artificial seams).

- **Thin-plate spline** (``CorrectionModel("tps", smoothing=s)``). A flexible
  *local* surface; only trustworthy near the control points, so it is applied
  with a support taper (1 inside the control hull, → 0 across a buffer ring).

Honesty guards:
- The base DTM is sampled only where it has data; nodata points don't constrain
  the fit (and are counted).
- Accuracy is judged by **leave-one-out CV** (`leave_one_out`), never the
  in-sample fit — a zero-smoothing TPS interpolates its own control exactly and
  would look perfect for the same circular reason the dGNSS-DTM does.
- A polynomial trend is *validated* only within the sampled region; applying it
  across the unsampled rest of the talar is a deliberate, documented
  extrapolation under the datum-effect assumption.

Depends on scipy + shapely (conda env, `environment.yml`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask

from .dtm import sample_raster_at_points, residual_stats, ResidualStats
from .io import resolve_path

__all__ = [
    "CorrectionModel",
    "ControlResiduals",
    "compute_residuals",
    "fit_corrector",
    "leave_one_out",
    "compare_models",
    "apply_correction",
]


# ---------------------------------------------------------------------------
# Correction model spec + correctors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionModel:
    kind: str            # "poly" | "tps"
    degree: int = 1      # poly only: 0=debias, 1=affine, 2=quadratic
    smoothing: float = 0.0  # tps only

    @property
    def is_global(self) -> bool:
        """Polynomial trends are global; TPS is local (needs a support taper)."""
        return self.kind == "poly"

    def label(self) -> str:
        if self.kind == "poly":
            return {0: "debias", 1: "affine", 2: "quadratic"}.get(self.degree, f"poly{self.degree}")
        return f"tps(s={self.smoothing:g})"


class _PolyCorrector:
    """Least-squares polynomial trend on mean-centred coords (for conditioning)."""

    def __init__(self, xy: np.ndarray, values: np.ndarray, degree: int):
        self.center = xy.mean(axis=0)
        self.degree = degree
        coef, *_ = np.linalg.lstsq(self._design(xy), values, rcond=None)
        self.coef = coef

    def _design(self, xy: np.ndarray) -> np.ndarray:
        c = np.atleast_2d(xy) - self.center
        x, y = c[:, 0], c[:, 1]
        cols = [np.ones(len(c))]
        if self.degree >= 1:
            cols += [x, y]
        if self.degree >= 2:
            cols += [x * x, x * y, y * y]
        return np.column_stack(cols)

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        return self._design(xy) @ self.coef


class _TPSCorrector:
    def __init__(self, xy: np.ndarray, values: np.ndarray, smoothing: float):
        from scipy.interpolate import RBFInterpolator
        self.rbf = RBFInterpolator(xy, values, kernel="thin_plate_spline",
                                   smoothing=smoothing)

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        return self.rbf(np.atleast_2d(xy))


def fit_corrector(xy: np.ndarray, values: np.ndarray, model: CorrectionModel):
    """Return a callable correction surface ``f(xy) -> correction values``."""
    if model.kind == "poly":
        return _PolyCorrector(xy, values, model.degree)
    if model.kind == "tps":
        return _TPSCorrector(xy, values, model.smoothing)
    raise ValueError(f"Unknown correction model kind: {model.kind!r}")


# ---------------------------------------------------------------------------
# Control residuals
# ---------------------------------------------------------------------------


@dataclass
class ControlResiduals:
    ids: np.ndarray       # (n,) point identifier (e.g. dGNSS Name), for cross-base alignment
    xy: np.ndarray        # (n, 2) projected coords of points with valid base
    base_z: np.ndarray    # (n,) base DTM elevation at those points
    ref_z: np.ndarray     # (n,) dGNSS elevation
    residual: np.ndarray  # (n,) ref_z - base_z (the correction to add)
    n_total: int
    n_valid: int

    @property
    def n_dropped(self) -> int:
        return self.n_total - self.n_valid


def compute_residuals(
    base_path: str | Path,
    gdf_points,
    *,
    z_col: str = "Elevation",
    id_col: str = "Name",
) -> ControlResiduals:
    """Sample ``base_path`` at each point; return residuals where the base is valid."""
    base_z_all = sample_raster_at_points(base_path, gdf_points)
    ref_all = gdf_points[z_col].to_numpy(dtype=float)
    xy_all = np.column_stack([[g.x for g in gdf_points.geometry],
                              [g.y for g in gdf_points.geometry]])
    ids_all = (gdf_points[id_col].to_numpy() if id_col in gdf_points.columns
               else np.arange(len(gdf_points)))

    valid = np.isfinite(base_z_all) & np.isfinite(ref_all)
    return ControlResiduals(
        ids=ids_all[valid],
        xy=xy_all[valid],
        base_z=base_z_all[valid],
        ref_z=ref_all[valid],
        residual=ref_all[valid] - base_z_all[valid],
        n_total=len(ref_all),
        n_valid=int(valid.sum()),
    )


# ---------------------------------------------------------------------------
# Leave-one-out CV
# ---------------------------------------------------------------------------


def leave_one_out(
    res: ControlResiduals,
    model: CorrectionModel,
) -> tuple[np.ndarray, ResidualStats, ResidualStats]:
    """Leave-one-out CV. Returns (loo_corrected_z, stats_corrected, stats_base).

    For each point: fit the correction from the others, predict it there, apply.
    The two returned stats share the same points and are directly comparable.
    """
    n = res.n_valid
    loo = np.empty(n, dtype=float)
    idx = np.arange(n)
    for i in range(n):
        m = idx != i
        corr = fit_corrector(res.xy[m], res.residual[m], model)
        loo[i] = res.base_z[i] + float(np.atleast_1d(corr(res.xy[i:i + 1]))[0])

    return loo, residual_stats(loo, res.ref_z), residual_stats(res.base_z, res.ref_z)


def compare_models(
    res: ControlResiduals,
    models: list[CorrectionModel],
    *,
    restrict_ids: np.ndarray | None = None,
) -> tuple["pd.DataFrame", dict[str, np.ndarray]]:
    """LOO every model; return a stats table + per-point LOO-corrected z arrays.

    ``restrict_ids`` scores only points whose id is in the set (for a fair
    common-subset comparison across bases with different coverage). The LOO
    *fit* still uses all of this base's control — only the scoring is restricted.
    """
    import pandas as pd

    if restrict_ids is not None:
        keep = np.isin(res.ids, restrict_ids)
    else:
        keep = np.ones(res.n_valid, dtype=bool)

    base_stats = residual_stats(res.base_z[keep], res.ref_z[keep])
    rows = [{"model": "base (raw)", "n": int(keep.sum()),
             "rmse": base_stats.rmse, "mae": base_stats.mae, "bias": base_stats.bias}]
    loo_by_model: dict[str, np.ndarray] = {}
    for model in models:
        loo, _, _ = leave_one_out(res, model)
        loo_by_model[model.label()] = loo
        s = residual_stats(loo[keep], res.ref_z[keep])
        rows.append({"model": model.label(), "n": int(keep.sum()),
                     "rmse": s.rmse, "mae": s.mae, "bias": s.bias})
    return pd.DataFrame(rows).set_index("model"), loo_by_model


# ---------------------------------------------------------------------------
# Apply the correction to the full raster
# ---------------------------------------------------------------------------


def _pixel_centers(transform, width: int, height: int):
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    return np.asarray(xs).reshape(height, width), np.asarray(ys).reshape(height, width)


def _support_weights(res: ControlResiduals, transform, width, height, buffer_m: float):
    """1 inside the control hull, linear taper → 0 over buffer_m, 0 beyond."""
    from scipy.ndimage import distance_transform_edt
    from shapely.geometry import MultiPoint

    hull = MultiPoint([tuple(p) for p in res.xy]).convex_hull
    inside = geometry_mask([hull], out_shape=(height, width),
                           transform=transform, invert=True)
    px = abs(transform.a)
    dist_m = distance_transform_edt(~inside) * px
    weight = np.where(inside, 1.0, np.clip(1.0 - dist_m / buffer_m, 0.0, 1.0))
    return weight.astype("float64")


@dataclass
class CorrectionResult:
    out_path: str
    model: str
    n_control: int
    tapered: bool
    corrected_min: float
    corrected_max: float
    max_abs_correction: float


def apply_correction(
    base_path: str | Path,
    res: ControlResiduals,
    out_path: str | Path,
    model: CorrectionModel,
    *,
    taper_buffer_m: float | None = None,
) -> CorrectionResult:
    """Fit the correction on all control residuals and write ``base + C`` to disk.

    ``taper_buffer_m``: if None, the correction is applied globally (correct for a
    polynomial trend). If a number, the correction is tapered to 0 beyond the
    control hull + that buffer (appropriate for a local TPS). Defaults to global
    for ``poly`` models and 25 m for ``tps`` models.
    """
    base_path = resolve_path(base_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if taper_buffer_m is None and not model.is_global:
        taper_buffer_m = 25.0  # sensible default for local TPS

    with rasterio.open(base_path) as src:
        profile = src.profile.copy()
        base = src.read(1).astype("float64")
        nodata = src.nodata
        transform = src.transform
        width, height = src.width, src.height

    valid_base = np.isfinite(base)
    if nodata is not None:
        valid_base &= base != nodata

    if taper_buffer_m is not None:
        weight = _support_weights(res, transform, width, height, taper_buffer_m)
        needs = (weight > 0) & valid_base
    else:
        weight = np.ones((height, width), dtype="float64")
        needs = valid_base

    correction = np.zeros((height, width), dtype="float64")
    if needs.any():
        X, Y = _pixel_centers(transform, width, height)
        corr = fit_corrector(res.xy, res.residual, model)
        correction[needs] = np.asarray(corr(np.column_stack([X[needs], Y[needs]])))

    corrected = base.copy()
    applied = weight * correction
    corrected[valid_base] = base[valid_base] + applied[valid_base]
    if nodata is not None:
        corrected[~valid_base] = nodata

    profile.update(dtype="float32", count=1, compress="deflate", predictor=3,
                   nodata=nodata if nodata is not None else -9999.0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(corrected.astype("float32"), 1)

    vals = corrected[valid_base]
    return CorrectionResult(
        out_path=str(out_path),
        model=model.label(),
        n_control=res.n_valid,
        tapered=taper_buffer_m is not None,
        corrected_min=float(vals.min()),
        corrected_max=float(vals.max()),
        max_abs_correction=float(np.abs(applied[valid_base]).max()) if vals.size else 0.0,
    )
