# drone-reserve

Drone RGB photogrammetry of the Reserva Natural Municipal del Pilar
(Buenos Aires, Argentina): structural and habitat characterization from a
single-campaign UAV flight.

This is the first of three remote-sensing portfolio projects. Goal: a
reproducible pipeline + technical report + interactive web map that a
reviewer can clone and re-run in under an hour.

## Zones

| Zone | Spanish name | Area    | Notes                                                          |
|------|--------------|---------|----------------------------------------------------------------|
| A    | *Talar*      | ~12 ha  | Dense *Celtis tala* forest on old tosqueras; wetland edge.    |
| B    | *Pastizal*   | ~13 ha  | Pampas grassland (*Cortaderia selloana* and others), low veg. |

## Data inventory (confirmed)

- **Acquisition:** DJI Phantom 4 Pro, RGB 20 MP, grid flights, May 2025.
- **Photogrammetry:** Pix4D → orthomosaic, DSM, DTM, dense point cloud.
- **CRS:** WGS 84 / UTM 21S — **EPSG:32721** (metric).
- **Ground truth:**
  - 42 dGNSS control points (terrain altitude + tagged tree positions).
  - 11 individuals with rangefinder height measurements, across 7 species:
    *Celtis tala*, *Phytolacca dioica*, *Solanum granulosum-leprosum*,
    *Salix humboldtiana*, *Morus* sp., *Baccharis* sp., *Cortaderia selloana*.

## Background — 2025 RAE poster (prior work)

This project builds on a poster presented at the I Reunión Trinacional de
Ecología (González Bianco, Alonzo, De Luca, Morandeira, 2025). The original
work was done mostly in QGIS with ad-hoc Python figures. See [`previuos/`](previuos/)
for the abstract, the poster PDF, and the three published figures.

**What the poster established:**

- VANT-DTM vs dGNSS-DTM, by zone:

  | Subset             | MAE  | MSE  | RMSE | R²   |
  |--------------------|------|------|------|------|
  | Full data          | 0.64 | 0.82 | 0.90 | 0.51 |
  | High tree density  | 1.12 | 1.63 | 1.28 | 0.05 |
  | Low tree density   | 0.21 | 0.08 | 0.28 | 0.31 |

- In the dense talar, **VANT-DTM systematically underestimates the ground**
  by ~1–2 m — the drone is reconstructing canopy where it should see soil.
- Both VANT-CHM and dGNSS-CHM correlate strongly with field heights
  (Pearson 0.95–0.98), but **VANT-CHM has larger bias for tall trees** in
  high-density zones — a direct consequence of the DTM error.
- Conclusion: dGNSS-DTM is preferable in dense forest; UAV photogrammetry
  is sufficient elsewhere.

**Gaps the poster acknowledges (or that this project should close):**

- Only Pix4D's built-in ground filter was used — no comparison with CSF,
  SMRF, or other point-cloud-based filters.
- n=11 measured trees is too small for per-species analysis.
- No habitat segmentation, no algorithmic individual-tree detection,
  no landscape metrics, no hydrological characterization.
- Workflow was not end-to-end reproducible (mixed QGIS + Python).

## What this project adds beyond the poster

Each item below maps to a pipeline step (see TODO list).

1. **Full Python, end-to-end reproducible.** Notebooks + a small `src/`
   package; every figure in the report comes from a notebook with explicit
   inputs and a deterministic output path. No QGIS in the critical path.
2. **Head-to-head DTM ground-filtering comparison** (Pix4D-DTM vs CSF vs
   SMRF, all validated against the 42 dGNSS) — directly attacks the
   R²=0.05 headline finding.
3. **A hybrid DTM** that splices dGNSS-derived ground (in dense canopy)
   into the best photogrammetric DTM (elsewhere), producing a single
   continuous "best available" DTM for the reserve.
4. **Habitat-aware CHM error model.** Re-run CHM accuracy stratified by
   habitat class and by continuous canopy density, not just the binary
   Alta/Baja split.
5. **Algorithmic individual-tree detection** on the CHM and/or point
   cloud, validated against the 11 measured individuals. Turns the
   sample of 11 into a population-level distribution for the talar.
6. **Habitat segmentation** (wetland, forest canopy, low vegetation,
   bare soil, water, trails) using an RGB-only model — new product
   relative to the poster.
7. **Landscape metrics per zone** (canopy cover, mean canopy height,
   wetland extent, edge density, fragmentation).
8. **Hydrological characterization** of the wetland (flow accumulation,
   depression storage, plausible inundation extents) from the DTM.
9. **Interactive web map** and a technical report PDF that compose all
   of the above into a portfolio-grade deliverable.

## Pipeline TODO

Each item maps to a numbered notebook in `notebooks/` and reusable code
in `src/drone_reserve/`. Tags: **[R]** reproduces a poster result,
**[X]** extends beyond the poster.

- [ ] **01 — Inventory & sanity check.** Open ortho, DSM, point cloud
      tile(s). Confirm CRS (expect EPSG:32721), extents, GSD, file sizes.
      Quick thumbnails. **[R/X]**
- [ ] **02 — Point cloud → DTM/DSM.** Run **CSF** and **SMRF** ground
      filters on the dense cloud; also ingest the existing Pix4D-DTM.
      Rasterize, fill, validate **all three** vertical accuracies against
      the 42 dGNSS points (RMSE, bias, residual plots, per-zone breakdown
      that reproduces the poster's table). **[R + X]**
- [ ] **03 — Hybrid DTM.** Build a continuous DTM that uses
      dGNSS-derived ground in dense canopy and the best photogrammetric
      DTM elsewhere. Re-validate. **[X]**
- [ ] **04 — CHM.** `DSM − DTM` for each DTM variant; denoise, mask
      negatives. Validate against the 11 measured trees per CHM variant;
      reproduce the poster's VANT-CHM and dGNSS-CHM scatter plots, then
      add the hybrid-CHM variant. **[R + X]**
- [ ] **05 — Habitat segmentation.** Semantic classes: wetland, forest
      canopy, low vegetation, bare soil, water, trails. RGB-only;
      tile-based inference. **[X]**
- [ ] **06 — Individual tree detection.** Local-maxima / watershed on
      CHM and/or point-cloud-based segmentation inside the talar; match
      to the 11 measured tree positions; report population-level height
      distribution. **[X]**
- [ ] **07 — Landscape metrics.** Canopy cover, mean canopy height,
      wetland extent, edge density, fragmentation, per zone. **[X]**
- [ ] **08 — Hydrology.** Flow accumulation, depression storage,
      probable inundation extents from the hybrid DTM (WhiteboxTools). **[X]**
- [ ] **09 — Web map.** Folium / leafmap deliverable with layers for
      ortho, CHM, habitat classes, tree crowns, hydrology. **[X]**
- [ ] **10 — Report.** Technical PDF: methods, validation, results,
      with the poster as the baseline comparison. **[X]**

## New analyses worth adding (kept short, only the ones that pay off)

These came out of reading the poster carefully. Each is in the TODO above;
the brief justification lives here so the rationale is visible.

- **Stratify CHM error by continuous canopy density**, not just the
  binary Alta/Baja split. With ~hundreds of detected trees from step 06,
  we can plot CHM RMSE as a function of local canopy cover (%) and find
  where photogrammetry stops being trustworthy. This is the natural
  extension of the poster's headline finding.
- **Per-habitat error tables** once segmentation is in place — far more
  useful for a conservation reader than two-zone splits.
- **A publishable single-product hybrid DTM**, not just the recommendation
  to "use dGNSS in dense forest." If the reserve uses this map, they'll
  want one raster, not two.

## What I deliberately won't chase

To keep the scope honest:

- **Multispectral / NDVI / NIR-derived indices** — single-campaign RGB
  only; nothing to compute them from.
- **Deep-learning species classification** — n=11 trees across 7 species
  is far below any defensible training set.
- **Multi-temporal change detection** — only one flight exists.
- **Refined allometry / DBH–height models** — n=11 again.

## Repo layout

```
.
├── data/         # raw / interim / processed — NOT tracked
├── notebooks/    # orchestration + figures
├── src/
│   └── drone_reserve/   # reusable Python package (logic lives here)
├── outputs/      # deterministic pipeline artifacts — NOT tracked
├── previuos/     # 2025 RAE poster materials (abstract, PDF, figures)
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

Heavier deps (PyTorch, segmentation_models_pytorch, torchgeo, PDAL
bindings) are intentionally **not** in `requirements.txt`. They get added
when the corresponding pipeline step lands, after deciding whether the step
runs locally (GTX 1650 Ti, 4 GB VRAM) or on Colab.

## Conventions

See `CLAUDE.md` (local only, gitignored) for the full collaboration spec.
Highlights:

- Always tile / stream large rasters — never load whole products into RAM.
- Validate against ground truth and print residuals / RMSE / bias.
- No silent fallbacks: missing inputs raise; pipeline never substitutes.
- Notebooks orchestrate, `src/` holds reusable logic.
- Ask before adding heavy dependencies.
