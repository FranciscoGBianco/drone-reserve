# CLAUDE.md — Drone Photogrammetry: Buenos Aires Reserve

> Save this file as `CLAUDE.md` at the root of your project repo. Claude Code reads it automatically at the start of every session.

## Project overview

I'm building a high-resolution structural and habitat characterization of a periurban natural reserve in Buenos Aires, Argentina, using drone RGB photogrammetry. The output should be (1) a clean, reproducible analysis pipeline, (2) a conservation-oriented technical report, and (3) an interactive web map. This is the first of three remote-sensing projects I'm using to build a portfolio.

The reserve contains two flown zones:
- **Zone A:** ~12 ha, denser vegetation, includes a low forest parcel and wetland transition.
- **Zone B:** ~13 ha, lower vegetation.

## Data inventory

- Single-campaign drone flight, **RGB only** (no multispectral, no NIR — so no NDVI from this dataset).
- Photogrammetric products to derive or already available: orthomosaic, DSM, dense point cloud, mesh.
- Ground reference:
  - A few dGNSS height control points.
  - A small set of tree locations with measured heights (for CHM validation).

## Goals for this project

1. Clean orthomosaic + DSM + DTM + CHM pipeline, validated against dGNSS and tree-height ground points.
2. Semantic segmentation of habitat classes: wetland, forest canopy, low vegetation, bare soil, water, trails.
3. Individual tree detection in the forest parcel, validated against located trees.
4. Habitat-level metrics: canopy cover, mean canopy height, wetland extent, edge density, fragmentation indices.
5. Hydrological characterization of the wetland (flow accumulation, depression storage, probable inundation extents) from the DEM.
6. Deliverable: a Jupyter-notebook-driven pipeline, a short technical report, and an interactive Folium/Leaflet web map.

## My hardware and environment

- **Local:** Dell G5500 (2020), Intel Core i7 10th gen, NVIDIA GTX 1650 Ti (4 GB VRAM).
- **Remote:** Google Colab (incl. Pro if needed), large storage, limited RAM.
- I work in **VS Code** and **Jupyter notebooks**.
- The drone outputs are large (multi-GB orthomosaic, hundreds of millions of points). **Always tile, never load full rasters into RAM.**

## Stack and conventions I prefer

- **Python**, modern (3.11+).
- Geospatial: `rasterio`, `rioxarray`, `geopandas`, `shapely`, `pyproj`, `whitebox` / `whiteboxtools`, `pdal`, `laspy`, `open3d`.
- ML: `pytorch`, `segmentation_models_pytorch`, `torchgeo` where it fits.
- Visualization: `folium`, `leafmap`, `matplotlib`, `contextily`.
- Project structure: `data/`, `notebooks/`, `src/`, `outputs/`, `reports/`. Raw data stays out of version control; use a `.gitignore` from the start.
- **Reproducibility matters:** every notebook documents its inputs and produces deterministic outputs into `outputs/`.

## Constraints and ground rules for Claude Code

1. **Ask before installing heavy dependencies.** I want to know what's being added and why.
2. **Prefer tiling and streaming over loading whole rasters.** My GPU has 4 GB VRAM and my RAM is finite. If a step needs Colab, say so explicitly.
3. **Validate against ground truth wherever possible.** dGNSS for DEM accuracy, tree heights for CHM accuracy. Print residuals, RMSE, and bias.
4. **Document assumptions inline.** When a parameter is chosen (e.g., CSF cloth resolution for ground filtering), explain the choice in a comment.
5. **One tool per task.** Don't introduce three libraries to do one thing.
6. **Write small, composable functions.** Notebooks orchestrate; logic lives in `src/`.
7. **No silent fallbacks.** If a file is missing or a step fails, raise; don't substitute defaults.

## First session — what I want to do today

1. Review the data inventory together: confirm what photogrammetric products I already have vs. need to generate.
2. Set up the project skeleton (`pyproject.toml` or `requirements.txt`, folder layout, `.gitignore`, README, this `CLAUDE.md`).
3. Build a first notebook that: loads the orthomosaic, loads the dense point cloud (or a tile), produces a quick visualization to confirm CRS, extent, and quality.
4. Decide together which CRS to standardize on (probably a local UTM zone — let's pick the right one for Buenos Aires).
5. Sketch the rest of the pipeline as a TODO list in the README.

## Open questions I want you to ask me when relevant

- Did the photogrammetry come from Pix4D, Metashape, OpenDroneMap, or something else? (Affects metadata.)
- What CRS are the products in?
- How many dGNSS points and how many measured trees, exactly?
- Do I want the final web map self-hosted or on a free service like GitHub Pages?

## What "done" looks like for this project

A repo I can point a hiring manager or collaborator at, containing: a clean pipeline that reproduces every figure, a technical report PDF, a deployed interactive map, and a LinkedIn-ready summary. Reviewer test: another remote-sensing person should be able to clone the repo and reproduce my results in under an hour.
