"""Step 08b — 3D inundation visualization + illustrative hydrodynamics.

Two distinct things, kept clearly separate:

A. **Progressive flood-stage inundation (defensible).** A 3D terrain from the
   corrected DTM with a rising flat water plane (a "bathtub" flood by river stage):
   cells flood where ground elevation < stage. This is geometry-only — no invented
   dynamics — and answers "if the floodplain rises to elevation W, what floods?".
   Rendered to PNG, an animated GIF (rising stage), and an interactive HTML for the
   web map (step 09).

B. **Illustrative overland flow (NOT a calibrated result).** Landlab's OverlandFlow
   shallow-water solver run with a *uniform synthetic rainfall* pulse. We have no
   rainfall / river-stage / inflow data, so this only illustrates how water would
   route over the measured terrain. Always label it illustrative.

PyVista renders off-screen (no window needed). z is exaggerated for display because
the floodplain relief is only a few metres.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .io import resolve_path

__all__ = [
    "load_terrain",
    "build_terrain_mesh",
    "build_water_mesh",
    "render_inundation",
    "flood_gif",
    "overland_flow_demo",
]


def load_terrain(dtm_path, *, downsample: int = 1):
    """Read a DTM and return (Xc, Yc, Z, nodata_mask). Z has NaN at nodata."""
    dtm_path = resolve_path(dtm_path)
    with rasterio.open(dtm_path) as src:
        Z = src.read(1).astype("float64")
        nd = src.nodata
        transform = src.transform
    if downsample > 1:
        Z = Z[::downsample, ::downsample]
        transform = transform * transform.scale(downsample, downsample)
    H, W = Z.shape
    # Pixel-centre coords as clean (H, W) arrays (north-up affine).
    xs = transform.c + (np.arange(W) + 0.5) * transform.a
    ys = transform.f + (np.arange(H) + 0.5) * transform.e
    Xc, Yc = np.meshgrid(xs, ys)
    nodata = ~np.isfinite(Z)
    if nd is not None:
        nodata |= (Z == nd)
    Z = np.where(nodata, np.nan, Z)
    return Xc, Yc, Z, nodata


def build_terrain_mesh(Xc, Yc, Z, *, exag: float = 5.0):
    """PyVista StructuredGrid of the terrain. NaN cells hidden via nan_opacity at render."""
    import pyvista as pv
    zfill = np.where(np.isfinite(Z), Z, np.nanmin(Z))  # geometry needs finite z
    grid = pv.StructuredGrid(Xc, Yc, zfill * exag)
    elev = Z.copy()  # scalar carries NaN so nodata can be hidden
    grid["elevation"] = elev.ravel(order="F")
    return grid


def build_water_mesh(Xc, Yc, Z, stage: float, *, exag: float = 5.0):
    """Flat water plane at ``stage``; depth scalar is NaN where not flooded (ground >= stage)."""
    import pyvista as pv
    flooded = np.isfinite(Z) & (Z < stage)
    zwater = np.where(flooded, stage, np.nanmin(Z))
    grid = pv.StructuredGrid(Xc, Yc, zwater * exag)
    depth = np.where(flooded, stage - Z, np.nan)
    grid["depth"] = depth.ravel(order="F")
    return grid


def _plot(terrain, water=None, *, title="", out_png=None, out_html=None,
          window=(1100, 850), camera_elev=35.0):
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=list(window))
    p.add_mesh(terrain, scalars="elevation", cmap="gist_earth",
               nan_opacity=0.0, show_scalar_bar=True, scalar_bar_args={"title": "elev (m)"})
    if water is not None:
        p.add_mesh(water, scalars="depth", cmap="Blues", nan_opacity=0.0,
                   opacity=0.55, show_scalar_bar=True, scalar_bar_args={"title": "depth (m)"})
    if title:
        p.add_text(title, font_size=10)
    p.view_isometric()
    p.camera.elevation = camera_elev
    if out_png:
        p.screenshot(str(out_png))
    if out_html:
        p.export_html(str(out_html))
    p.close()


def render_inundation(dtm_path, stage, out_png=None, out_html=None, *,
                      exag: float = 5.0, downsample: int = 2, title: str | None = None):
    """Render terrain + a single flood stage to PNG and/or interactive HTML."""
    Xc, Yc, Z, _ = load_terrain(dtm_path, downsample=downsample)
    terrain = build_terrain_mesh(Xc, Yc, Z, exag=exag)
    water = build_water_mesh(Xc, Yc, Z, stage, exag=exag)
    if title is None:
        title = f"Flood stage = {stage:.2f} m  (z x{exag:g})"
    _plot(terrain, water, title=title, out_png=out_png, out_html=out_html)
    n_flood = int(np.sum(np.isfinite(Z) & (Z < stage)))
    px = abs(Xc[0, 1] - Xc[0, 0]) * abs(Yc[1, 0] - Yc[0, 0])
    return {"stage": stage, "flooded_ha": round(n_flood * px / 1e4, 2)}


def flood_gif(dtm_path, stages, out_gif, *, exag: float = 5.0, downsample: int = 2):
    """Animated GIF of rising flood stage over the 3D terrain."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    Xc, Yc, Z, _ = load_terrain(dtm_path, downsample=downsample)
    terrain = build_terrain_mesh(Xc, Yc, Z, exag=exag)
    p = pv.Plotter(off_screen=True, window_size=[1000, 800])
    p.add_mesh(terrain, scalars="elevation", cmap="gist_earth", nan_opacity=0.0)
    p.view_isometric(); p.camera.elevation = 35.0
    p.open_gif(str(out_gif))
    txt = p.add_text("", font_size=10)
    water_actor = None
    for s in stages:
        if water_actor is not None:
            p.remove_actor(water_actor)
        w = build_water_mesh(Xc, Yc, Z, s, exag=exag)
        water_actor = p.add_mesh(w, scalars="depth", cmap="Blues", nan_opacity=0.0,
                                 opacity=0.55, show_scalar_bar=False)
        p.remove_actor(txt); txt = p.add_text(f"stage = {s:.2f} m", font_size=10)
        p.write_frame()
    p.close()
    return str(out_gif)


def overland_flow_demo(dtm_path, out_dir, *, rain_mm_hr: float = 50.0,
                       duration_min: float = 30.0, dt_s: float = 2.0,
                       downsample: int = 2, exag: float = 5.0):
    """ILLUSTRATIVE Landlab OverlandFlow under uniform synthetic rainfall.

    No rainfall/inflow data exist for this site, so this only shows how water would
    route over the measured terrain — not a calibrated flood prediction. Returns the
    final water-depth array + renders a 3D snapshot and a 2D depth map.
    """
    from landlab import RasterModelGrid
    from landlab.components import OverlandFlow
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Xc, Yc, Z, nodata = load_terrain(dtm_path, downsample=downsample)
    H, W = Z.shape
    spacing = float(abs(Xc[0, 1] - Xc[0, 0]))

    # Landlab grid; fill nodata with a high wall and mark those nodes closed.
    z_fill = np.where(np.isfinite(Z), Z, np.nanmax(Z) + 50.0)
    grid = RasterModelGrid((H, W), xy_spacing=spacing)
    # Landlab node order is row-major from bottom-left; our array is top-left origin.
    grid.add_field("topographic__elevation", z_fill[::-1, :].ravel(), at="node")
    grid.add_zeros("surface_water__depth", at="node")
    # Close nodata nodes so flow ignores them; keep perimeter open as outlets.
    closed = nodata[::-1, :].ravel()
    grid.status_at_node[closed] = grid.BC_NODE_IS_CLOSED

    of = OverlandFlow(grid, steep_slopes=True)
    rain_ms = rain_mm_hr / 1000.0 / 3600.0  # mm/hr -> m/s
    elapsed, end = 0.0, duration_min * 60.0
    while elapsed < end:
        of.dt = dt_s
        grid.at_node["surface_water__depth"] += rain_ms * dt_s
        of.run_one_step(dt=dt_s)
        elapsed += dt_s

    depth = grid.at_node["surface_water__depth"].reshape(H, W)[::-1, :]
    depth = np.where(nodata, np.nan, depth)

    # 2D depth map
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.where(depth > 0.01, depth, np.nan), cmap="Blues", vmin=0, vmax=0.5)
    ax.set_title(f"ILLUSTRATIVE overland flow\n{rain_mm_hr:.0f} mm/hr for {duration_min:.0f} min")
    fig.colorbar(im, ax=ax, shrink=0.7, label="water depth (m)"); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(out_dir / "overland_flow_depth.png", dpi=140); plt.close(fig)

    # 3D snapshot: depth draped on terrain
    import pyvista as pv
    pv.OFF_SCREEN = True
    terrain = build_terrain_mesh(Xc, Yc, Z, exag=exag)
    wet = np.where(np.isfinite(Z) & (depth > 0.01), Z + depth, np.nanmin(Z))
    wgrid = pv.StructuredGrid(Xc, Yc, wet * exag)
    wgrid["depth"] = np.where(depth > 0.01, depth, np.nan).ravel(order="F")
    _plot(terrain, wgrid, title=f"ILLUSTRATIVE overland flow ({rain_mm_hr:.0f} mm/hr)",
          out_png=str(out_dir / "overland_flow_3d.png"))

    return {"max_depth_m": float(np.nanmax(depth)),
            "wet_area_ha": round(float(np.nansum(depth > 0.01)) * spacing * spacing / 1e4, 2),
            "depth_png": str(out_dir / "overland_flow_depth.png"),
            "scene_png": str(out_dir / "overland_flow_3d.png")}
