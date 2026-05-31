"""Reusable helpers for 3D / orthomosaic visualization.

Logic extracted from ``notebooks/visualize_orthomosaic.py`` so notebooks
and CLI scripts can compose it (per project rule: notebooks orchestrate,
logic lives in ``src/``).

Optional GPU acceleration via CuPy (array math) and Open3D's CUDA t-geometry
backend. CuPy is imported lazily inside :func:`activate_gpu`; Open3D is a
hard import (it's in ``requirements.txt``).

GPU acceleration coverage
-------------------------
    Stage                           CPU path        GPU path (after activate_gpu())
    ─────────────────────────────── ──────────────  ────────────────────────────────
    Coordinate transform math       NumPy           CuPy (on-device)
    NoData / alpha masking          NumPy boolean   CuPy boolean mask
    Colour normalisation            NumPy           CuPy
    Centring & z-scaling            NumPy           CuPy
    Voxel downsampling              Open3D CPU      Open3D CUDA PointCloud (t-geometry)
    Normal estimation               Open3D CPU KD   Open3D CUDA KD-tree

Caveat
------
:func:`load_rasters` reads each input fully into memory before sub-sampling.
That's fine for the quick-look viewer this module supports; the analysis
pipeline should use windowed reads instead (see the project rule on tiling).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d
import rasterio
import rasterio.transform

from .io import resolve_path

__all__ = [
    "GPU",
    "activate_gpu",
    "load_rasters",
    "build_point_cloud",
    "build_mesh",
    "launch_viewer",
]


# ---------------------------------------------------------------------------
# GPU bootstrap (lazy-imports CuPy so CPU-only users are unaffected)
# ---------------------------------------------------------------------------


class _GPUContext:
    """Holds the active array backend (NumPy or CuPy) and the Open3D-CUDA flag.

    Stays in NumPy/CPU mode by default. Call :func:`activate_gpu` (or
    :meth:`activate` directly) to opt in.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.xp = np            # array module: numpy or cupy
        self.device_id = 0
        self.o3d_cuda = False   # True when open3d.core.Device("CUDA:N") works

    def activate(self, device_id: int = 0) -> None:
        """Try to enable GPU; warn and fall back to CPU on any failure."""
        self.device_id = device_id

        # 1. CuPy ----------------------------------------------------------
        try:
            import cupy as cp

            cp.cuda.Device(device_id).use()
            _ = cp.array([1.0])  # warm-up; surfaces driver errors early
            self.xp = cp
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            name = (
                props["name"].decode()
                if isinstance(props["name"], bytes)
                else props["name"]
            )
            mem_gb = props["totalGlobalMem"] / 1024**3
            print(f"  CuPy      : ✓  {name}  ({mem_gb:.1f} GB VRAM)  [device {device_id}]")
        except ImportError:
            print("  CuPy      : ✗  not installed — pip install cupy-cuda12x")
            print("              Falling back to NumPy for array math.")
        except Exception as e:
            print(f"  CuPy      : ✗  {e}")
            print("              Falling back to NumPy for array math.")

        # 2. Open3D CUDA t-geometry ---------------------------------------
        # Probe with a real allocation; PointCloud() succeeds even on CPU-only wheels.
        try:
            _dev = o3d.core.Device(f"CUDA:{device_id}")
            _probe = o3d.core.Tensor([1.0], device=_dev)
            del _probe
            self.o3d_cuda = True
            print(f"  Open3D    : ✓  CUDA t-geometry available  [device {device_id}]")
        except Exception as e:
            self.o3d_cuda = False
            if "BUILD_CUDA_MODULE" in str(e) or "Unsupported device" in str(e):
                print("  Open3D    : ✗  CPU-only wheel — install open3d-gpu or build with CUDA")
            else:
                print(f"  Open3D    : ✗  {e}")
            print("              Using CPU geometry ops for normals / voxel downsampling.")

        self.enabled = True


# Module-level singleton — pragmatic given that Open3D viewers are global anyway.
GPU = _GPUContext()


def activate_gpu(device_id: int = 0) -> _GPUContext:
    """Enable GPU acceleration for subsequent calls. Returns the context."""
    GPU.activate(device_id)
    return GPU


# ---------------------------------------------------------------------------
# Internal array helpers
# ---------------------------------------------------------------------------


def _to_numpy(arr) -> np.ndarray:
    """Return a NumPy array regardless of whether ``arr`` is CuPy or NumPy."""
    if type(arr).__module__ == "cupy":
        return arr.get()
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Public: raster loading
# ---------------------------------------------------------------------------


def load_rasters(
    ortho_path: str | Path,
    dem_path: str | Path,
    downsample: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load and align an orthomosaic (RGB/RGBA) and a DEM raster.

    Heavy array math runs on GPU when :func:`activate_gpu` has been called.

    Parameters
    ----------
    ortho_path
        Path to the orthomosaic GeoTIFF (RGB or RGBA).
    dem_path
        Path to the DEM GeoTIFF (single-band elevation).
    downsample
        Pixel stride. ``downsample=5`` keeps every 5th pixel in each axis.

    Returns
    -------
    xyz : (N, 3) float64
        Centred X/Y plus elevation Z.
    colors : (N, 3) float64
        RGB in ``[0, 1]``.
    meta : dict
        Keys: ``crs``, ``centroid_x``, ``centroid_y``, ``z_min``, ``z_max``,
        ``n_points``.

    Raises
    ------
    FileNotFoundError
        If either path doesn't exist (per project rule: no silent fallbacks).
    """
    ortho_path = resolve_path(ortho_path)
    dem_path = resolve_path(dem_path)

    xp = GPU.xp  # numpy or cupy

    # 1. Read rasters (always on CPU / host memory via Rasterio) ----------
    with rasterio.open(dem_path) as dem_src:
        dem_full_np = dem_src.read(1).astype(np.float64)
        dem_nodata = dem_src.nodata
        dem_transform = dem_src.transform
        dem_crs = dem_src.crs

    with rasterio.open(ortho_path) as ortho_src:
        n_bands = ortho_src.count
        rgb_np = ortho_src.read([1, 2, 3]).astype(np.float64)
        alpha_np = ortho_src.read(4).astype(np.float64) if n_bands >= 4 else None

    if rgb_np.shape[1:] != dem_full_np.shape:
        raise ValueError(
            "Orthomosaic and DEM grids must match. "
            f"Ortho: {rgb_np.shape[1:]}, DEM: {dem_full_np.shape}. "
            "Reproject / resample one to the other before calling load_rasters."
        )

    # 2. Stride sub-sample (slicing is trivial on CPU) --------------------
    sy = slice(0, dem_full_np.shape[0], downsample)
    sx = slice(0, dem_full_np.shape[1], downsample)

    dem_sub_np = dem_full_np[sy, sx]
    rgb_sub_np = rgb_np[:, sy, sx]
    alpha_np = alpha_np[sy, sx] if alpha_np is not None else None

    # 3. Pixel → world coordinates (Rasterio must run on CPU) -------------
    rows_idx, cols_idx = np.indices(dem_sub_np.shape)
    rows_orig = rows_idx * downsample
    cols_orig = cols_idx * downsample

    xs, ys = rasterio.transform.xy(dem_transform, rows_orig.ravel(), cols_orig.ravel())
    x_np = np.array(xs, dtype=np.float64)
    y_np = np.array(ys, dtype=np.float64)

    # 4. Upload to GPU (if available) -------------------------------------
    t0 = time.perf_counter()
    x = xp.asarray(x_np)
    y = xp.asarray(y_np)
    z = xp.asarray(dem_sub_np.ravel())
    rgb = xp.asarray(rgb_sub_np)
    alph = xp.asarray(alpha_np.ravel()) if alpha_np is not None else None

    if GPU.enabled and GPU.xp is not np:
        print(f"  H→D xfer  : {time.perf_counter() - t0:.3f}s")

    # 5. Build validity mask ----------------------------------------------
    valid = ~xp.isnan(z)
    if dem_nodata is not None:
        valid &= ~xp.isclose(z, float(dem_nodata), atol=1e-3)
    else:
        # Fallback sentinel range — common DEM nodata is -9999.
        valid &= z > -9998.0

    if alph is not None:
        valid &= alph > 0

    # 6. Mask, centre, normalise ------------------------------------------
    x_v = x[valid]
    y_v = y[valid]
    z_v = z[valid]

    cx = float(_to_numpy(x_v.mean()))
    cy = float(_to_numpy(y_v.mean()))
    x_v = x_v - cx
    y_v = y_v - cy

    r = rgb[0].ravel()[valid]
    g = rgb[1].ravel()[valid]
    b = rgb[2].ravel()[valid]

    raw_max = float(_to_numpy(rgb.max()))
    bit_scale = 255.0 if raw_max <= 255.0 else 65535.0

    # 7. Transfer back to NumPy for Open3D --------------------------------
    t1 = time.perf_counter()
    xyz_np = np.column_stack([_to_numpy(x_v), _to_numpy(y_v), _to_numpy(z_v)])
    colors_np = np.column_stack([_to_numpy(r), _to_numpy(g), _to_numpy(b)]) / bit_scale

    if GPU.enabled and GPU.xp is not np:
        print(f"  D→H xfer  : {time.perf_counter() - t1:.3f}s")

    meta = {
        "crs": str(dem_crs),
        "centroid_x": cx,
        "centroid_y": cy,
        "z_min": float(xyz_np[:, 2].min()),
        "z_max": float(xyz_np[:, 2].max()),
        "n_points": int(
            valid.sum() if type(valid).__module__ == "numpy" else _to_numpy(valid).sum()
        ),
    }
    return xyz_np, colors_np, meta


# ---------------------------------------------------------------------------
# Internal: Open3D geometry helpers
# ---------------------------------------------------------------------------


def _voxel_downsample_cpu(pcd: o3d.geometry.PointCloud, voxel_size: float):
    return pcd.voxel_down_sample(voxel_size)


def _voxel_downsample_gpu(pcd_legacy, voxel_size: float, device_id: int):
    """Open3D t-geometry CUDA path; converts back to legacy PointCloud."""
    dev = o3d.core.Device(f"CUDA:{device_id}")
    pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd_legacy, device=dev)
    pcd_t_ds = pcd_t.voxel_down_sample(voxel_size)
    return pcd_t_ds.to_legacy()


def _estimate_normals_gpu(pcd_legacy, radius: float, max_nn: int, device_id: int):
    """Normal estimation via Open3D CUDA t-geometry."""
    dev = o3d.core.Device(f"CUDA:{device_id}")
    pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd_legacy, device=dev)
    pcd_t.estimate_normals(max_nn=max_nn, radius=radius)
    return pcd_t.to_legacy()


# ---------------------------------------------------------------------------
# Public: geometry construction & viewer
# ---------------------------------------------------------------------------


def build_point_cloud(
    xyz: np.ndarray,
    colors: np.ndarray,
    z_scale: float = 1.0,
    voxel_size: float = 0.0,
) -> o3d.geometry.PointCloud:
    """Build a coloured Open3D point cloud with optional voxel downsampling.

    ``voxel_size`` is in map units (metres if the source raster is in UTM).
    Normals are estimated with a 20 m / 30-neighbour hybrid search and
    oriented toward a point well above the scene.
    """
    pts = xyz.copy()
    pts[:, 2] *= z_scale

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    if voxel_size > 0:
        print(f"  Voxel downsampling at {voxel_size:.2f} m …")
        t = time.perf_counter()
        if GPU.o3d_cuda:
            pcd = _voxel_downsample_gpu(pcd, voxel_size, GPU.device_id)
        else:
            pcd = _voxel_downsample_cpu(pcd, voxel_size)
        print(f"    → {len(pcd.points):,} pts  ({time.perf_counter() - t:.2f}s)")

    print("  Estimating normals …")
    t = time.perf_counter()
    if GPU.o3d_cuda:
        pcd = _estimate_normals_gpu(pcd, radius=20, max_nn=30, device_id=GPU.device_id)
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30)
        )
    pcd.orient_normals_towards_camera_location([0, 0, 1e6])
    print(f"    done  ({time.perf_counter() - t:.2f}s)")
    return pcd


def build_mesh(
    xyz: np.ndarray,
    colors: np.ndarray,
    z_scale: float = 1.0,
) -> o3d.geometry.TriangleMesh:
    """Poisson reconstruction from a coloured point cloud, with density trim."""
    pcd = build_point_cloud(xyz, colors, z_scale, voxel_size=0)
    print("  Running Poisson mesh reconstruction …")
    t = time.perf_counter()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False
    )
    threshold = np.quantile(np.asarray(densities), 0.05)
    mesh.remove_vertices_by_mask(np.asarray(densities) < threshold)
    mesh.compute_vertex_normals()
    print(f"    done  ({time.perf_counter() - t:.2f}s)")

    # Transfer colours from nearest point-cloud point.
    kdt = o3d.geometry.KDTreeFlann(pcd)
    vtx = np.asarray(mesh.vertices)
    vtx_colors = np.zeros((len(vtx), 3))
    pcd_colors = np.asarray(pcd.colors)
    for i, v in enumerate(vtx):
        _, idx, _ = kdt.search_knn_vector_3d(v, 1)
        vtx_colors[i] = pcd_colors[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vtx_colors)
    return mesh


def launch_viewer(
    geometry,
    point_size: float = 2.0,
    bg: Sequence[float] = (0.10, 0.10, 0.12),
    title: str = "3D Orthomosaic",
) -> None:
    """Open an interactive Open3D viewer window (blocks until closed)."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(geometry)

    ropt = vis.get_render_option()
    ropt.background_color = np.array(bg)
    ropt.point_size = point_size
    ropt.light_on = True

    vis.get_view_control().set_zoom(0.6)

    print("\n  Controls: Left-drag = rotate | Right-drag = pan | Scroll = zoom | Q = quit\n")
    vis.run()
    vis.destroy_window()
