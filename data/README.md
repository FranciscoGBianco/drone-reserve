# data/

Local-only. Not tracked in git (see root `.gitignore`).

Suggested layout:

```
data/
  raw/            # untouched drone exports (ortho, DSM, point cloud, mesh)
    zone_a/
    zone_b/
  interim/        # cleaned / reprojected / tiled intermediates
  processed/      # analysis-ready rasters & vectors (DTM, CHM, segmentation)
  ground_truth/
    dgnss/        # dGNSS control points (CSV / shp / gpkg)
    trees/        # measured tree locations + heights
  external/       # basemaps, official reserve boundary, etc.
```

Each subfolder should contain its own short `README.md` documenting
provenance (source, date, CRS, units, processing software).
