# drone-reserve

Drone RGB photogrammetry of the Reserva Natural Municipal del Pilar
(Buenos Aires, Argentina): structural and habitat characterization from a
single-campaign UAV flight.

This is the first of three remote-sensing portfolio projects. Goal: a
reproducible pipeline + technical report + interactive web map that a
reviewer can clone and re-run in under an hour.

## Zones

| Zone | Area    | Notes                                          |
|------|---------|------------------------------------------------|
| A    | ~12 ha  | Denser vegetation; low forest + wetland edge.  |
| B    | ~13 ha  | Lower vegetation.                              |

## Repo layout

```
.
├── data/         # raw / interim / processed — NOT tracked
├── notebooks/    # orchestration + figures
├── src/
│   └── drone_reserve/   # reusable Python package (logic lives here)
├── outputs/      # deterministic pipeline artifacts — NOT tracked
├── reports/      # technical report sources + final PDFs
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # PowerShell on Windows
pip install -r requirements.txt
```

Heavier deps (PyTorch, segmentation_models_pytorch, torchgeo, Open3D, PDAL
bindings) are intentionally **not** in `requirements.txt`. They get added
when the corresponding pipeline step lands, after deciding whether the step
runs locally (GTX 1650 Ti, 4 GB VRAM) or on Colab.

## Pipeline TODO

Each item maps to a numbered notebook in `notebooks/` and reusable code in
`src/drone_reserve/`.

- [ ] **01 — Inventory & sanity check.** Open ortho, DSM, point cloud
      tile(s). Print CRS, extent, GSD, file sizes. Quick thumbnails.
      Decide canonical CRS (likely a local UTM zone for Buenos Aires).
- [ ] **02 — Point cloud → DTM/DSM.** Ground filtering (CSF or SMRF),
      rasterize, fill, validate vertical accuracy against dGNSS points
      (RMSE, bias, residual plots).
- [ ] **03 — CHM.** `DSM − DTM`, denoise, mask negatives. Validate against
      measured tree heights.
- [ ] **04 — Habitat segmentation.** Semantic classes: wetland, forest
      canopy, low vegetation, bare soil, water, trails. RGB-only model;
      tile-based inference.
- [ ] **05 — Individual tree detection.** Local-maxima / watershed on CHM
      inside the forest parcel; match to measured tree locations.
- [ ] **06 — Landscape metrics.** Canopy cover, mean canopy height,
      wetland extent, edge density, fragmentation indices, per-zone.
- [ ] **07 — Hydrology.** Flow accumulation, depression storage,
      probable inundation extents from the DTM (WhiteboxTools).
- [ ] **08 — Web map.** Folium / leafmap deliverable with layers for
      ortho, CHM, habitat classes, tree crowns, hydrology.
- [ ] **09 — Report.** Technical PDF: methods, validation, results.
