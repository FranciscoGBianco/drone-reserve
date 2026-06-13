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
   R²=0.05 headline finding. **Result:** SMRF more than halves Pix4D's
   dense-canopy error (RMSE 1.25 m → 0.58 m; overall 0.90 m → 0.43 m).
3. **A bias-corrected SMRF DTM** that uses the 42 dGNSS points as *control*
   (not as a wholesale surface) to model and remove SMRF's residual bias,
   producing a single continuous, ground-truth-pinned DTM — with SMRF-alone
   kept as the honest baseline and leave-one-out CV deciding which ships.
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

- [X] **01 — Inventory & sanity check.** Open ortho, DSM, point cloud
      tile(s). Confirm CRS (expect EPSG:32721), extents, GSD, file sizes.
      Quick thumbnails. **[R/X]**
- [X] **02 — Point cloud → DTM/DSM.** Ran **CSF** and **SMRF** ground
      filters (PDAL) on the dense cloud + ingested the Pix4D-DTM and
      dGNSS-DTM. Validated against the 42 dGNSS points, reproduced the
      poster's Pix4D row (MAE/RMSE to 2 dp), added a continuous
      canopy-density vs residual analysis + coverage / common-subset
      comparison. **Outcome:** SMRF (RMSE 0.43 m, 0.58 m dense) and a
      retuned CSF (more accurate at control points, lower coverage) both
      beat Pix4D; the two are close, so **both** go into step 03 and
      leave-one-out CV picks the production DTM. dGNSS-DTM excluded from
      accuracy claims (circular — interpolated from the same points). **[R + X]**
- [X] **03 — dGNSS bias-corrected DTM (talar).** Sampled base-DTM residuals at
      the 42 dGNSS points; the residual proved to be a **constant + tilt** (datum
      discrepancy), so the correction is a global **affine trend**, not a local
      surface (LOO showed TPS overfits and loses to the trend). Applied to both
      bases; validated leave-one-out: SMRF 0.43→0.31 m, CSF-retuned 0.31→0.22 m
      (~30% each), bias zeroed. **Canonical = SMRF + affine**
      (`outputs/03_corrected/talar_dtm_corrected_0p5m.tif`) for coverage;
      CSF-retuned+affine kept as cross-check. Gain concentrated in dense canopy
      (Alta 0.58→0.38) with a small Baja cost. DTMs **gap-filled to ~100%**
      (fill only empty cells → RMSE 0.43→0.44, vs 0.61 if widening the IDW
      radius) with a distance-to-measured **confidence layer**; CHM-ready.
      Pastizal = SMRF-raw + fill, uncorrected (no dGNSS there). **[X]**
- [X] **04 — CHM.** `DSM − DTM` per variant (DSM bilinear-resampled to the
      0.5 m grid via streamed WarpedVRT; negatives clamped). Validated vs the 11
      field tree heights (crown-max): **Corrected CHM best — r=0.97, RMSE 0.87 m,
      bias +0.19 m**, beating raw VANT (r=0.93, RMSE 1.99 m, bias +1.30 m) and
      dGNSS (r=0.95). The step-03 correction cuts tall-tree bias +1.3→+0.2 m.
      Reproduces the poster + improves it. Tree trunk positions needed QGIS QC
      (dGNSS error under canopy; two were swapped). **[R + X]**
- [~] **05 — Habitat segmentation (unsupervised baseline).** KMeans on
      **ExG + brightness + CHM** (RGB-only, no NDVI — height is the key axis),
      clusters named by a transparent centroid rule; forest canopy pinned to
      measured CHM ≥ 3 m. Reliable 4-class set (forest canopy, low vegetation,
      bare/dry ground, water). Areas: talar 2.4/3.6/5.9 ha, pastizal 2.0/5.9/5.5;
      10/11 field trees land in vegetation classes. No water present; **wetland &
      trails not separable in RGB** (deferred). Preliminary (no accuracy figure) —
      upgrade path is a supervised Random Forest with QGIS labels. **[X]**
- [X] **06 — Individual tree detection.** CHM local-maxima with height-scaled
      allometric NMS (Popescu-style; scipy only). **Detection 9/11** of the field
      trees (2 misses are the lowest open shrubs), **matched-height RMSE 0.83 m,
      bias +0.14 m** — both robust. Extends the poster's n=11 to a forest-wide
      height distribution (mean ~5 m, tail to ~17 m). Density ≈400 trees/ha but
      **parameter-sensitive (range 305–808)** — reported as indicative, not measured.
      Crown delineation (watershed) deferred (would add scikit-image). **[X]**
- [X] **07 — Landscape metrics.** Per-zone canopy (cover %, mean/p95/max height)
      + FRAGSTATS-style fragmentation (NP, PD, MPS, LPI, edge density) via
      scipy.ndimage, with a **25 m² MMU sieve** so per-pixel noise doesn't inflate
      patch counts. **Talar forest consolidated** (cover ≥3 m 20%, LPI 5.2%) vs
      **pastizal forest scattered** (15%, LPI 1.8%) in a grassland matrix.
      Wetland extent **not reported** (not separable in RGB). **[X]**
- [ ] **08 — Hydrology.** Flow accumulation, depression storage,
      probable inundation extents from the corrected DTM (WhiteboxTools). **[X]**
- [ ] **09 — Web map.** Folium / leafmap deliverable with layers for
      ortho, CHM, habitat classes, tree crowns, hydrology. **[X]**
- [ ] **10 — Report.** Technical PDF: methods, validation, results,
      with the poster as the baseline comparison. **[X]**

## Limitations

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
├── requirements.txt    # pip / venv path
├── environment.yml     # conda env (PDAL stack)
├── pyproject.toml      # editable install of the drone_reserve package
├── .gitignore
└── README.md
```

## Setup

There are two supported environments. **Both** require the editable install
of this package (`pip install -e .`) so notebooks can `import drone_reserve`
without `sys.path` shims.

### Option A — pip venv (good for steps 01 / inventory + viewer)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # PowerShell on Windows
pip install -r requirements.txt
pip install -e .
```

### Option B — conda env (required from step 02 onwards — PDAL)

PDAL has no Windows wheels on PyPI, so pip-installing it is fragile. Use the
conda spec instead — it bundles the PDAL C++ library and the `python-pdal`
bindings together:

```powershell
conda env create -f environment.yml
conda activate drone-reserve
python -m ipykernel install --user --name drone-reserve --display-name "Python (drone-reserve)"
```

`environment.yml` already runs `pip install -e .` for you, so the package is
importable inside the env.

Heavier deps (PyTorch, segmentation_models_pytorch, torchgeo) are still
intentionally **not** in either env. They get added when the corresponding
pipeline step lands, after deciding whether the step runs locally
(GTX 1650 Ti, 4 GB VRAM) or on Colab.