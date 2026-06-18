"""Step 08 — hydrological characterization from the DTM (WhiteboxTools).

Pipeline (each WhiteboxTools step reads/writes a GeoTIFF):

1. **Fill depressions** -> hydrologically filled DEM. The *filled minus original*
   difference is the **depression-storage** map: how deep water can pond in each
   sink before it overflows. On this floodplain that's the wetland-ponding signal
   (the talar's old tosqueras are real depressions).
2. **Breach depressions (least-cost)** -> a flow-routable DEM that carves drains
   through dams instead of flooding them (gentler on real terrain) — used only for
   flow routing, not for storage.
3. **D8 flow pointer + flow accumulation** -> upslope contributing area per cell;
   high accumulation traces drainage lines.
4. **Extract streams** at an accumulation threshold -> channel network.
5. **Stochastic depression analysis** -> probability each cell lies in a depression
   given DEM error (Monte-Carlo perturbation by ``rmse``); a *probable inundation*
   map that honestly propagates the DTM's vertical uncertainty.

Terrain here is a near-flat floodplain, so the depression/ponding outputs are far
more meaningful than a crisp drainage network — interpret accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from .io import resolve_path

__all__ = ["make_wbt", "depression_storage", "run_hydrology", "HydroResult"]


def make_wbt(work_dir: str | Path):
    """Return a quiet WhiteboxTools set to ``work_dir`` (created if needed)."""
    import whitebox

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(str(work_dir))
    wbt.set_verbose_mode(False)
    return wbt


def prep_dem_for_wbt(dtm_path, out_path):
    """Re-write a DEM as a WhiteboxTools-readable GeoTIFF.

    WhiteboxTools' GeoTIFF reader cannot decode the floating-point predictor
    (PREDICTOR=3) we use elsewhere for compression, so we re-encode without it.
    Values are unchanged.
    """
    dtm_path = resolve_path(dtm_path)
    with rasterio.open(dtm_path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
    # Strip every compression-related key and write a plain, uncompressed GeoTIFF —
    # WBT chokes on DEFLATE+PREDICTOR=3 and even on a re-encoded predictor, so the
    # robust choice is no compression at all (the DTM is small).
    for k in ("compress", "predictor", "zlevel", "interleave"):
        profile.pop(k, None)
    profile.update(driver="GTiff", tiled=False)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
    return str(out_path)


def depression_storage(filled_path, original_path, out_path):
    """Write (filled - original) = depression depth per cell; return (volume_m3, max_depth_m).

    Volume = sum(depth * cell_area). Cells with no storage are 0; nodata preserved.
    """
    filled_path = resolve_path(filled_path)
    original_path = resolve_path(original_path)
    with rasterio.open(filled_path) as f:
        filled = f.read(1).astype("float64"); profile = f.profile.copy()
        nd = f.nodata; px = abs(f.transform.a) * abs(f.transform.e)
    with rasterio.open(original_path) as o:
        orig = o.read(1).astype("float64"); ond = o.nodata

    valid = np.isfinite(filled) & np.isfinite(orig)
    if nd is not None:
        valid &= filled != nd
    if ond is not None:
        valid &= orig != ond

    depth = np.where(valid, np.maximum(filled - orig, 0.0), nd if nd is not None else -9999.0)
    profile.update(dtype="float32", count=1, nodata=nd if nd is not None else -9999.0,
                   compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(depth.astype("float32"), 1)

    d = (filled - orig)[valid]
    d = np.maximum(d, 0.0)
    return float((d * px).sum()), float(d.max() if d.size else 0.0)


@dataclass
class HydroResult:
    out_dir: str
    filled: str
    storage: str
    breached: str
    flow_accum: str
    streams: str
    inundation_prob: str
    storage_volume_m3: float
    max_depth_m: float
    inundation_area_ha_p50: float   # area with P(depression) >= 0.5
    zone_area_ha: float


def run_hydrology(
    dtm_path: str | Path,
    out_dir: str | Path,
    *,
    name: str = "zone",
    stream_threshold_cells: int = 2000,
    stoch_rmse: float = 0.3,
    stoch_range_m: float = 10.0,
    stoch_iterations: int = 100,
) -> HydroResult:
    """Run the full DTM-hydrology pipeline for one zone. Returns paths + summary stats."""
    dtm_path = resolve_path(dtm_path)
    out_dir = Path(out_dir)
    wbt = make_wbt(out_dir)

    # WhiteboxTools is run with basenames relative to the working dir. Build a
    # (basename, abspath) pair for each product; use basenames in wbt calls and
    # abspaths for rasterio reads / existence checks.
    bn = {k: f"{name}_{k}.tif" for k in
          ["dem", "filled", "storage", "breached", "d8ptr", "accum", "streams", "inund"]}
    ap = {k: str(out_dir / v) for k, v in bn.items()}

    def _check(key):
        # wbt returns 0 even on a Rust panic, so verify the output really exists.
        if not Path(ap[key]).exists():
            raise RuntimeError(f"WhiteboxTools did not produce {bn[key]} "
                               f"(run with set_verbose_mode(True) to see why).")

    # 0. Re-encode the DEM to a WBT-readable GeoTIFF (no float predictor).
    prep_dem_for_wbt(dtm_path, ap["dem"])

    # 1. Fill depressions -> storage map.
    wbt.fill_depressions(bn["dem"], bn["filled"], fix_flats=True); _check("filled")
    vol, maxd = depression_storage(ap["filled"], ap["dem"], ap["storage"])

    # 2. Breach depressions (Lindsay 2016) for flow routing; fill any residual pits.
    #    (The least-cost variant panics on this near-flat DEM, so we use the standard one.)
    wbt.breach_depressions(bn["dem"], bn["breached"], fill_pits=True)
    _check("breached")

    # 3. D8 pointer + flow accumulation on the breached DEM.
    wbt.d8_pointer(bn["breached"], bn["d8ptr"]); _check("d8ptr")
    wbt.d8_flow_accumulation(bn["breached"], bn["accum"], out_type="cells"); _check("accum")

    # 4. Extract streams at an accumulation threshold.
    wbt.extract_streams(bn["accum"], bn["streams"], threshold=stream_threshold_cells)
    _check("streams")

    # 5. Stochastic depression analysis -> P(in depression) given DEM error.
    wbt.stochastic_depression_analysis(bn["dem"], bn["inund"], rmse=stoch_rmse,
                                       range=stoch_range_m, iterations=stoch_iterations)
    _check("inund")
    p = ap  # keep the rest of the function (paths + stats) unchanged

    # Summary stats.
    with rasterio.open(dtm_path) as src:
        a = src.read(1); nd = src.nodata; pxa = abs(src.transform.a) * abs(src.transform.e)
        valid = np.isfinite(a) & ((a != nd) if nd is not None else True)
        zone_area_ha = float(valid.sum()) * pxa / 1e4
    with rasterio.open(p["inund"]) as src:
        prob = src.read(1); pnd = src.nodata
        pv = np.isfinite(prob) & ((prob != pnd) if pnd is not None else True)
        inund_ha = float(((prob >= 0.5) & pv).sum()) * pxa / 1e4

    return HydroResult(
        out_dir=str(out_dir), filled=p["filled"], storage=p["storage"],
        breached=p["breached"], flow_accum=p["accum"], streams=p["streams"],
        inundation_prob=p["inund"], storage_volume_m3=round(vol, 1),
        max_depth_m=round(maxd, 2), inundation_area_ha_p50=round(inund_ha, 2),
        zone_area_ha=round(zone_area_ha, 2),
    )
