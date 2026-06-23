"""Step 09 — interactive web map assembly.

Builds a self-contained, GitHub-Pages-ready static site from the pipeline's
canonical rasters/vectors. Notebooks orchestrate; this module holds the logic.

Geometry choices (so layers line up correctly on a Leaflet/web-mercator map):

- **Small derived rasters are reprojected to EPSG:3857** (web mercator) and saved
  as RGBA PNGs, then placed with their 3857 extent converted to 4326 corner
  bounds. A 3857 image stretched linearly between 3857-derived corners is
  geometrically exact on a web-mercator map; a naive 4326 ``ImageOverlay`` would
  smear by the equirectangular-vs-mercator difference. These rasters are 0.5 m,
  sub-MB, so a single PNG overlay each is cheap.
- **The orthomosaic is XYZ-tiled** with ``gdal2tiles`` (mercator profile) rather
  than inlined — it is multi-hundred-MB at ~3.4 cm GSD. Tiling honours the
  project rule "tile, never load full rasters into RAM" and keeps the map
  responsive. The 12-13 ha extents make the pyramid small (tens of MB).

Colours/scales live here once so the map legend and the report agree. Class
colours mirror ``segment.CLASS_NAMES`` (0 forest, 1 low veg, 2 bare/dry,
3 water); keep them in sync if the segmentation classes change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from .io import resolve_path

__all__ = [
    "WEBMERC", "WGS84", "HABITAT_COLORS", "HABITAT_NAMES",
    "reproject_band", "colorize_continuous", "colorize_categorical",
    "save_png", "overlay_bounds_4326", "make_overlay",
    "vector_to_geojson", "tile_orthomosaic",
    "new_map", "add_tiles", "add_image_overlay", "add_geojson_points",
    "add_zone_outlines", "add_legend", "add_float_gif", "finalize",
    "write_index_html",
]

WEBMERC = "EPSG:3857"
WGS84 = "EPSG:4326"

# Habitat class -> (name, RGB). Mirrors segment.CLASS_NAMES; nodata is 255.
HABITAT_NAMES = {0: "Forest canopy", 1: "Low vegetation", 2: "Bare / dry ground", 3: "Water"}
HABITAT_COLORS = {
    0: (27, 120, 55),    # forest — dark green
    1: (166, 217, 106),  # low veg — light green
    2: (191, 163, 124),  # bare/dry — tan
    3: (44, 123, 182),   # water — blue
}


# --------------------------------------------------------------------------- #
# Raster -> reprojected RGBA PNG overlay
# --------------------------------------------------------------------------- #
def reproject_band(src_path, *, band: int = 1, resampling: Resampling = Resampling.bilinear,
                   dst_crs: str = WEBMERC):
    """Read one band reprojected to ``dst_crs`` via a WarpedVRT (no full-res load).

    Returns ``(arr, mask, bounds)`` where ``arr`` is float64, ``mask`` is True at
    nodata, and ``bounds`` is ``(left, bottom, right, top)`` in ``dst_crs``. GDAL
    excludes the source nodata from resampling, so nodata stays nodata exactly.
    """
    src_path = resolve_path(src_path)
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        with WarpedVRT(src, crs=dst_crs, resampling=resampling) as vrt:
            arr = vrt.read(band).astype("float64")
            nodata = vrt.nodata if vrt.nodata is not None else src_nodata
            bounds = tuple(vrt.bounds)  # left, bottom, right, top
    mask = ~np.isfinite(arr)
    if nodata is not None:
        mask |= np.isclose(arr, nodata)
    return arr, mask, bounds


def colorize_continuous(arr, mask, *, cmap: str = "viridis",
                        vmin: float | None = None, vmax: float | None = None):
    """Map a float array through a matplotlib colormap to RGBA uint8 (nodata -> transparent)."""
    import matplotlib
    from matplotlib.colors import Normalize

    valid = arr[~mask]
    if vmin is None:
        vmin = float(np.nanmin(valid)) if valid.size else 0.0
    if vmax is None:
        vmax = float(np.nanmax(valid)) if valid.size else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = matplotlib.colormaps[cmap](norm(np.where(mask, vmin, arr)))
    rgba = (rgba * 255).astype("uint8")
    rgba[mask] = (0, 0, 0, 0)
    return rgba, (vmin, vmax)


def colorize_categorical(arr, mask, color_map: dict[int, tuple[int, int, int]]):
    """Map integer class codes to RGBA uint8 using ``color_map`` (nodata -> transparent)."""
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")
    ai = arr.astype("int64")
    for val, (r, g, b) in color_map.items():
        m = (~mask) & (ai == val)
        rgba[m] = (r, g, b, 255)
    return rgba


def save_png(rgba, out_path) -> str:
    """Write an (H, W, 4) uint8 array to a PNG."""
    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, rgba)
    return str(out_path)


def overlay_bounds_4326(bounds_3857):
    """3857 ``(left, bottom, right, top)`` -> folium bounds ``[[S, W], [N, E]]`` in 4326."""
    w, s, e, n = transform_bounds(WEBMERC, WGS84, *bounds_3857)
    return [[s, w], [n, e]]


def make_overlay(src_path, out_png, *, kind: str, cmap: str = "viridis",
                 vmin: float | None = None, vmax: float | None = None,
                 color_map: dict | None = None):
    """Reproject + colorize a raster to a web-mercator RGBA PNG overlay.

    ``kind`` is ``"continuous"`` (uses ``cmap``/``vmin``/``vmax``, bilinear) or
    ``"categorical"`` (uses ``color_map``, nearest). Returns a dict with the PNG
    path, folium ``bounds``, and the value range (continuous only) for the legend.
    """
    if kind == "categorical":
        arr, mask, b3857 = reproject_band(src_path, resampling=Resampling.nearest)
        rgba = colorize_categorical(arr, mask, color_map or HABITAT_COLORS)
        rng = None
    elif kind == "continuous":
        arr, mask, b3857 = reproject_band(src_path, resampling=Resampling.bilinear)
        rgba, rng = colorize_continuous(arr, mask, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        raise ValueError(f"kind must be 'continuous' or 'categorical', got {kind!r}")
    save_png(rgba, out_png)
    return {"png": str(out_png), "bounds": overlay_bounds_4326(b3857), "range": rng}


# --------------------------------------------------------------------------- #
# Vector -> GeoJSON (always WGS84 for Leaflet)
# --------------------------------------------------------------------------- #
def vector_to_geojson(src, out_path, *, columns: list[str] | None = None,
                      encoding: str | None = None) -> str:
    """Reproject a vector to WGS84 and write GeoJSON. Optionally keep only ``columns``."""
    import geopandas as gpd

    g = gpd.read_file(resolve_path(src), encoding=encoding)
    g = g.to_crs(WGS84)
    if columns is not None:
        keep = [c for c in columns if c in g.columns] + ["geometry"]
        g = g[keep]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(out_path, driver="GeoJSON")
    return str(out_path)


# --------------------------------------------------------------------------- #
# Orthomosaic -> XYZ tile pyramid
# --------------------------------------------------------------------------- #
def tile_orthomosaic(src_path, out_dir, *, zoom: str = "14-21",
                     resampling: str = "average") -> str:
    """XYZ-tile an orthomosaic into ``out_dir`` with gdal2tiles (mercator, --xyz).

    ``--xyz`` gives standard slippy-map ``{z}/{x}/{y}.png`` so a folium TileLayer
    works without TMS y-flipping. ``-w none`` skips the bundled viewer HTML (we
    embed the tiles in our own map). The ortho's alpha band drives transparency.
    """
    from osgeo_utils import gdal2tiles

    src_path = resolve_path(src_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = ["gdal2tiles", "--profile=mercator", "--xyz", f"--zoom={zoom}",
            f"--resampling={resampling}", "--webviewer=none", "--quiet",
            str(src_path), str(out_dir)]
    gdal2tiles.main(argv)
    if not (out_dir / "tilemapresource.xml").exists() and not any(out_dir.glob("*/")):
        raise RuntimeError(f"gdal2tiles produced no tiles in {out_dir}")
    return str(out_dir)


# --------------------------------------------------------------------------- #
# Folium map assembly
# --------------------------------------------------------------------------- #
def new_map(center, *, zoom_start: int = 16):
    """A folium Map with satellite + street basemaps (satellite default)."""
    import folium

    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles=None,
                   control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite (Esri)", overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street (OSM)", overlay=False, control=True).add_to(m)
    return m


def add_tiles(m, name, tiles_rel, *, attr: str = "drone-reserve", show: bool = True,
              max_zoom: int = 23):
    """Add a local XYZ tile layer (relative URL, e.g. ``tiles/talar_ortho/{z}/{x}/{y}.png``)."""
    import folium

    folium.TileLayer(tiles=tiles_rel + "/{z}/{x}/{y}.png", attr=attr, name=name,
                     overlay=True, control=True, show=show, max_zoom=max_zoom,
                     max_native_zoom=21, tms=False).add_to(m)
    return m


def add_image_overlay(m, name, png_rel, bounds, *, opacity: float = 0.85, show: bool = False):
    """Add a reprojected PNG overlay referenced by **relative URL** (not embedded).

    folium base64-embeds any local image path it can open, which would bloat the
    saved HTML by megabytes. We seed the overlay with a 1x1 transparent pixel to
    satisfy the constructor, then point its URL at the external PNG so the site
    loads the file at runtime (keeps ``map.html`` small and the PNGs cacheable).
    """
    import folium

    io = folium.raster_layers.ImageOverlay(
        image=np.zeros((1, 1, 4), dtype="uint8"), bounds=bounds, opacity=opacity,
        name=name, overlay=True, control=True, show=show, cross_origin=False,
    )
    io.url = png_rel  # rendered into L.imageOverlay("<url>", ...)
    io.add_to(m)
    return m


def add_geojson_points(m, name, geojson_rel, *, value_field: str = "height",
                       palette: str = "YlOrRd", show: bool = True):
    """Add detected treetops as circle markers sized + coloured by ``value_field``.

    Reads the GeoJSON client-side via folium.GeoJson with a style function baked
    from the data range, so it stays a single static file.
    """
    import json
    import folium
    import matplotlib
    from matplotlib.colors import Normalize, to_hex

    with open(geojson_rel, encoding="utf-8") as f:
        gj = json.load(f)
    vals = [ft["properties"].get(value_field) for ft in gj["features"]
            if ft["properties"].get(value_field) is not None]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps[palette]

    fg = folium.FeatureGroup(name=name, show=show)
    for ft in gj["features"]:
        lon, lat = ft["geometry"]["coordinates"]
        v = ft["properties"].get(value_field)
        v = float(v) if v is not None else vmin
        folium.CircleMarker(
            location=[lat, lon], radius=2 + 4 * norm(v),
            color=None, fill=True, fill_color=to_hex(cmap(norm(v))), fill_opacity=0.85,
            popup=f"{value_field}: {v:.1f} m",
        ).add_to(fg)
    fg.add_to(m)
    return m


def add_zone_outlines(m, geojson_rel, *, name: str = "Flight footprints"):
    """Add zone footprint polygons as hollow outlines."""
    import folium

    folium.GeoJson(
        geojson_rel, name=name,
        style_function=lambda _f: {"color": "#ffffff", "weight": 2, "fill": False,
                                   "dashArray": "5,5"},
    ).add_to(m)
    return m


def _cmap_css(cmap_name: str, n: int = 8) -> str:
    """A CSS ``linear-gradient(...)`` string sampling a matplotlib colormap."""
    import matplotlib
    from matplotlib.colors import to_hex

    cmap = matplotlib.colormaps[cmap_name]
    stops = [to_hex(cmap(i / (n - 1))) for i in range(n)]
    return "linear-gradient(to right, " + ", ".join(stops) + ")"


def add_legend(m, entries):
    """Add a single collapsible legend panel (bottom-right).

    ``entries`` is a list of dicts: continuous ``{"type":"ramp","label","cmap",
    "vmin","vmax","unit"}`` or categorical ``{"type":"cats","label","items":[(name,
    (r,g,b)), ...]}``.
    """
    import folium

    rows = []
    for e in entries:
        if e["type"] == "ramp":
            rows.append(
                f'<div class="lg-row"><div class="lg-lab">{e["label"]}</div>'
                f'<div class="lg-bar" style="background:{_cmap_css(e["cmap"])}"></div>'
                f'<div class="lg-tick"><span>{e["vmin"]:g}</span>'
                f'<span>{e["vmax"]:g} {e.get("unit","")}</span></div></div>'
            )
        else:
            sw = "".join(
                f'<div class="lg-sw"><span style="background:rgb{tuple(c)}"></span>{nm}</div>'
                for nm, c in e["items"]
            )
            rows.append(f'<div class="lg-row"><div class="lg-lab">{e["label"]}</div>{sw}</div>')
    html = f"""
<style>
 #legend {{ position:fixed; bottom:18px; right:12px; z-index:9999; background:rgba(255,255,255,.94);
   border:1px solid #bbb; border-radius:8px; font:12px/1.3 system-ui,sans-serif; color:#222;
   max-width:230px; box-shadow:0 1px 6px rgba(0,0,0,.3); }}
 #legend summary {{ cursor:pointer; padding:7px 10px; font-weight:600; list-style:none; }}
 #legend .lg-body {{ padding:4px 10px 10px; }}
 #legend .lg-row {{ margin:8px 0; }}
 #legend .lg-lab {{ font-weight:600; margin-bottom:3px; }}
 #legend .lg-bar {{ height:11px; border-radius:3px; border:1px solid #999; }}
 #legend .lg-tick {{ display:flex; justify-content:space-between; font-size:11px; color:#555; }}
 #legend .lg-sw {{ display:flex; align-items:center; gap:6px; margin:2px 0; }}
 #legend .lg-sw span {{ width:14px; height:14px; border-radius:3px; border:1px solid #777; display:inline-block; }}
</style>
<details id="legend" open><summary>Legend</summary><div class="lg-body">{''.join(rows)}</div></details>
"""
    m.get_root().html.add_child(folium.Element(html))
    return m


def add_float_gif(m, gif_rel, *, bottom: int = 18, left: int = 12, width: int = 200):
    """Float the rising-flood GIF on the map as a teaser linking to the 3D scene."""
    import folium

    html = f"""
<div style="position:fixed; bottom:{bottom}px; left:{left}px; z-index:9999;
     background:rgba(255,255,255,.94); border:1px solid #bbb; border-radius:8px;
     padding:6px; box-shadow:0 1px 6px rgba(0,0,0,.3); font:11px system-ui,sans-serif;">
  <div style="font-weight:600; margin-bottom:4px;">Flood stage (3D &mdash; talar)</div>
  <img src="{gif_rel}" width="{width}" style="border-radius:4px; display:block;">
  <div style="color:#555; margin-top:3px;">Rising river-stage bathtub model</div>
</div>
"""
    m.get_root().html.add_child(folium.Element(html))
    return m


def finalize(m, *, fit_bounds=None):
    """Add the layer control and optionally fit to bounds ([[S,W],[N,E]])."""
    import folium

    folium.LayerControl(collapsed=False).add_to(m)
    if fit_bounds is not None:
        m.fit_bounds(fit_bounds)
    return m


# --------------------------------------------------------------------------- #
# Landing page
# --------------------------------------------------------------------------- #
def write_index_html(out_dir, *, map_rel: str, flood_html_rel: str, gif_rel: str,
                     title: str, subtitle: str, stats: dict | None = None) -> str:
    """Write the portfolio landing page: intro, embedded map, 3D flood-sim panel.

    The flood simulation is given top billing (per the brief): an interactive 3D
    river-stage scene alongside the rising-flood GIF, with the defensible-vs-
    illustrative caveat spelled out.
    """
    out_dir = Path(out_dir)
    stat_html = ""
    if stats:
        cells = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>'
                        for k, v in stats.items())
        stat_html = f'<div class="stats">{cells}</div>'
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
 :root {{ --fg:#1c2b22; --muted:#5a6b60; --accent:#1b7837; --bg:#f7f9f7; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; color:var(--fg);
   background:var(--bg); }}
 header {{ background:linear-gradient(135deg,#16432a,#2f7d4f); color:#fff; padding:28px 22px; }}
 header h1 {{ margin:0 0 4px; font-size:1.7rem; }}
 header p {{ margin:0; opacity:.9; }}
 main {{ max-width:1100px; margin:0 auto; padding:22px; }}
 section {{ margin:26px 0; }}
 h2 {{ font-size:1.25rem; color:var(--accent); border-bottom:2px solid #d6e6da; padding-bottom:6px; }}
 iframe {{ width:100%; border:1px solid #cdd8d0; border-radius:10px; background:#fff; }}
 .map-frame {{ height:74vh; min-height:520px; }}
 .flood-grid {{ display:grid; grid-template-columns:1.5fr 1fr; gap:16px; align-items:start; }}
 .flood-grid iframe {{ height:60vh; min-height:420px; }}
 .flood-side img {{ width:100%; border-radius:10px; border:1px solid #cdd8d0; }}
 .note {{ background:#fff7e6; border:1px solid #f0d9a8; border-radius:8px; padding:10px 12px;
   font-size:.9rem; color:#6b5a2a; }}
 .stats {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:14px; }}
 .stat {{ background:#fff; border:1px solid #d6e6da; border-radius:10px; padding:10px 16px; }}
 .stat b {{ display:block; font-size:1.35rem; color:var(--accent); }}
 .stat span {{ font-size:.8rem; color:var(--muted); }}
 footer {{ color:var(--muted); font-size:.85rem; padding:22px; text-align:center; }}
 @media (max-width:820px) {{ .flood-grid {{ grid-template-columns:1fr; }} }}
</style></head>
<body>
<header><h1>{title}</h1><p>{subtitle}</p></header>
<main>
  <section>
    <p>Interactive structural &amp; habitat characterization of a periurban
    floodplain reserve from a single RGB drone campaign (Pix4D, May 2025;
    EPSG:32721). Toggle layers in the map: orthomosaic, canopy height (CHM),
    habitat classes, detected trees, and the hydrology / flood layers.</p>
    {stat_html}
  </section>
  <section>
    <h2>Interactive map</h2>
    <iframe class="map-frame" src="{map_rel}" loading="lazy" title="Interactive map"></iframe>
  </section>
  <section>
    <h2>Flood simulation (3D)</h2>
    <div class="flood-grid">
      <iframe src="{flood_html_rel}" loading="lazy" title="3D inundation"></iframe>
      <div class="flood-side">
        <img src="{gif_rel}" alt="Rising flood-stage animation">
        <p class="note"><b>Defensible (river-stage bathtub):</b> the 3D scene and
        animation flood the measured terrain as a flat water surface rises to a
        given river stage &mdash; geometry only, no invented dynamics. The map's
        <i>inundation probability</i> and <i>depression storage</i> layers come
        from the DTM hydrology (Step 08). Overland-flow hydrodynamics under
        synthetic rainfall are kept <i>illustrative only</i> (no site rainfall
        data) and are not shown as a prediction.</p>
      </div>
    </div>
  </section>
</main>
<footer>drone-reserve &middot; reproducible pipeline (Steps 01&ndash;09) &middot;
vertical exaggeration applied in 3D for visibility.</footer>
</body></html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    return str(out_dir / "index.html")
