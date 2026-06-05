"""Metadata-only inventory probes for the drone-reserve dataset.

Every function here reads only headers / sidecar metadata — never a full
raster or point cloud into memory (per project rule: tile, don't load).

Returns dataclasses + dicts so notebooks can render them however they want.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import fiona
import laspy
import rasterio

from .io import resolve_path

__all__ = [
    "RasterInfo",
    "LasInfo",
    "VectorLayerInfo",
    "probe_raster",
    "probe_las",
    "probe_vector",
    "scan_directory",
    "format_size",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def format_size(n_bytes: int) -> str:
    """Human-readable byte count (KB / MB / GB, base-1024)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n_bytes)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:,.1f} {u}"
        size /= 1024
    return f"{size:,.1f} TB"  # unreachable


# ---------------------------------------------------------------------------
# Raster probe
# ---------------------------------------------------------------------------


@dataclass
class RasterInfo:
    path: str
    file_bytes: int
    driver: str
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    crs: str | None
    bounds: tuple[float, float, float, float]  # left, bottom, right, top
    res_x: float
    res_y: float
    nodata: float | None
    has_overviews: bool
    overview_levels: tuple[int, ...]
    transform_units: str  # "metre" / "degree" / "unknown"

    @property
    def gsd_m(self) -> float | None:
        """Ground sample distance in metres, if the CRS is projected metric."""
        if self.transform_units != "metre":
            return None
        return (abs(self.res_x) + abs(self.res_y)) / 2

    @property
    def extent_ha(self) -> float | None:
        """Footprint area in hectares, if CRS is metric."""
        if self.transform_units != "metre":
            return None
        left, bottom, right, top = self.bounds
        return abs(right - left) * abs(top - bottom) / 10_000


def probe_raster(path: str | Path) -> RasterInfo:
    """Open the raster as metadata-only and return a :class:`RasterInfo`."""
    p = resolve_path(path)
    file_bytes = p.stat().st_size
    with rasterio.open(p) as src:
        ovr = src.overviews(1) if src.count else []
        crs = src.crs
        units = "unknown"
        if crs is not None:
            try:
                unit_name = crs.linear_units or ""
                if unit_name.lower() in ("metre", "meter"):
                    units = "metre"
                elif unit_name.lower() == "degree":
                    units = "degree"
            except Exception:
                pass
            if units == "unknown" and crs.is_geographic:
                units = "degree"
            if units == "unknown" and crs.is_projected:
                units = "metre"
        return RasterInfo(
            path=str(p),
            file_bytes=file_bytes,
            driver=src.driver,
            width=src.width,
            height=src.height,
            count=src.count,
            dtypes=tuple(src.dtypes),
            crs=str(crs) if crs else None,
            bounds=tuple(src.bounds),
            res_x=src.res[0],
            res_y=src.res[1],
            nodata=src.nodata,
            has_overviews=bool(ovr),
            overview_levels=tuple(ovr),
            transform_units=units,
        )


# ---------------------------------------------------------------------------
# LAS / LAZ probe
# ---------------------------------------------------------------------------


@dataclass
class LasInfo:
    path: str
    file_bytes: int
    version: str
    point_format: int
    n_points: int
    scales: tuple[float, float, float]
    offsets: tuple[float, float, float]
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]
    crs: str | None
    classifications_present: tuple[int, ...] | None
    # classifications_present is None when we did not scan classes
    # (scanning means reading point data; we keep header-only by default)


def probe_las(path: str | Path, *, scan_classes: bool = False) -> LasInfo:
    """Read LAS/LAZ header only; optionally scan classifications.

    ``scan_classes=True`` triggers a full point read just for the classification
    column — only enable for small tiles or when you actually need it.
    """
    p = resolve_path(path)
    file_bytes = p.stat().st_size

    classes: tuple[int, ...] | None = None

    with laspy.open(str(p)) as reader:
        h = reader.header
        version = f"{h.version.major}.{h.version.minor}"
        crs = None
        try:
            wkt = h.parse_crs()
            crs = str(wkt) if wkt else None
        except Exception:
            crs = None

        info = LasInfo(
            path=str(p),
            file_bytes=file_bytes,
            version=version,
            point_format=int(h.point_format.id),
            n_points=int(h.point_count),
            scales=tuple(float(s) for s in h.scales),
            offsets=tuple(float(o) for o in h.offsets),
            mins=tuple(float(v) for v in h.mins),
            maxs=tuple(float(v) for v in h.maxs),
            crs=crs,
            classifications_present=None,
        )

        if scan_classes:
            unique: set[int] = set()
            for pts in reader.chunk_iterator(1_000_000):
                unique.update(int(c) for c in set(pts.classification))
            info.classifications_present = tuple(sorted(unique))

    return info


# ---------------------------------------------------------------------------
# Vector / OGR probe
# ---------------------------------------------------------------------------


@dataclass
class VectorLayerInfo:
    path: str
    layer: str
    driver: str
    geometry_type: str | None
    crs: str | None
    feature_count: int
    field_names: tuple[str, ...]


def probe_vector(path: str | Path) -> list[VectorLayerInfo]:
    """Return one :class:`VectorLayerInfo` per layer (GeoPackages can carry many)."""
    p = resolve_path(path)
    out: list[VectorLayerInfo] = []
    for layer_name in fiona.listlayers(str(p)):
        with fiona.open(str(p), layer=layer_name) as src:
            schema = src.schema or {}
            out.append(
                VectorLayerInfo(
                    path=str(p),
                    layer=layer_name,
                    driver=src.driver,
                    geometry_type=schema.get("geometry"),
                    crs=str(src.crs) if src.crs else None,
                    feature_count=len(src),
                    field_names=tuple((schema.get("properties") or {}).keys()),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------


@dataclass
class DirectoryScan:
    rasters: list[RasterInfo] = field(default_factory=list)
    las_files: list[LasInfo] = field(default_factory=list)
    vectors: list[VectorLayerInfo] = field(default_factory=list)
    image_counts: dict[str, int] = field(default_factory=dict)  # dir -> JPG count


_RASTER_EXTS = {".tif", ".tiff"}
_LAS_EXTS = {".las", ".laz"}
_VECTOR_EXTS = {".gpkg", ".shp", ".geojson", ".kml", ".gml"}
_IMAGE_EXTS = {".jpg", ".jpeg"}


def scan_directory(
    root: str | Path,
    *,
    follow_symlinks: bool = False,
    image_dirs_only_counts: bool = True,
) -> DirectoryScan:
    """Walk ``root`` and probe every raster / LAS / vector file we recognise.

    Raw drone imagery folders are counted (file count per directory) rather
    than enumerated — there are typically hundreds of JPGs per flight.
    """
    root = resolve_path(root)
    scan = DirectoryScan()

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()

        if ext in _RASTER_EXTS:
            try:
                scan.rasters.append(probe_raster(path))
            except Exception as e:
                print(f"  ! raster probe failed for {path}: {e}")
        elif ext in _LAS_EXTS:
            try:
                scan.las_files.append(probe_las(path))
            except Exception as e:
                print(f"  ! LAS probe failed for {path}: {e}")
        elif ext in _VECTOR_EXTS:
            try:
                scan.vectors.extend(probe_vector(path))
            except Exception as e:
                print(f"  ! vector probe failed for {path}: {e}")
        elif ext in _IMAGE_EXTS and image_dirs_only_counts:
            scan.image_counts[str(path.parent)] = (
                scan.image_counts.get(str(path.parent), 0) + 1
            )

    return scan


# ---------------------------------------------------------------------------
# Pretty-printer (kept simple; notebooks can format their own way)
# ---------------------------------------------------------------------------


def _trunc(path: str, root: str) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def print_summary(scan: DirectoryScan, root: str | Path) -> None:
    root = str(Path(root).resolve())
    print(f"\n=== Rasters ({len(scan.rasters)}) ===")
    for r in scan.rasters:
        crs_short = (r.crs or "—").split('"')[1] if r.crs and '"' in r.crs else (r.crs or "—")
        gsd = f"GSD={r.gsd_m:.3f} m" if r.gsd_m else f"res=({r.res_x:g}, {r.res_y:g})"
        ext = f"{r.extent_ha:.2f} ha" if r.extent_ha else "(non-metric CRS)"
        print(
            f"  {_trunc(r.path, root)}\n"
            f"    size={format_size(r.file_bytes)}  {r.width}x{r.height}px  "
            f"bands={r.count} dtype={r.dtypes[0]} nodata={r.nodata}  "
            f"{gsd}  {ext}\n"
            f"    crs={crs_short}  overviews={r.overview_levels or 'none'}"
        )

    print(f"\n=== Point clouds ({len(scan.las_files)}) ===")
    for L in scan.las_files:
        print(
            f"  {_trunc(L.path, root)}\n"
            f"    size={format_size(L.file_bytes)}  v{L.version}  pf={L.point_format}  "
            f"n_points={L.n_points:,}\n"
            f"    x:[{L.mins[0]:.2f}, {L.maxs[0]:.2f}]  y:[{L.mins[1]:.2f}, {L.maxs[1]:.2f}]  "
            f"z:[{L.mins[2]:.2f}, {L.maxs[2]:.2f}]"
        )

    print(f"\n=== Vector layers ({len(scan.vectors)}) ===")
    for v in scan.vectors:
        print(
            f"  {_trunc(v.path, root)} :: {v.layer}\n"
            f"    driver={v.driver}  geom={v.geometry_type}  features={v.feature_count}\n"
            f"    crs={v.crs}  fields={list(v.field_names)}"
        )

    print(f"\n=== Raw imagery folders ({len(scan.image_counts)}) ===")
    for d, n in sorted(scan.image_counts.items()):
        print(f"  {_trunc(d, root)} :: {n} JPGs")
