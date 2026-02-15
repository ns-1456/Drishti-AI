# Drishti-AI Notebooks

Run from project root so `src` is importable, or run from this folder (notebooks set `ROOT` automatically).

## 1. `01_project_overview_and_architecture.ipynb`

- Four phases (Sky, Street, Breath, Mind)
- **Delhi NCR ROI map** with city markers
- **Data flow diagram** (offline/online pipelines → model → prediction)
- Tech stack and repo structure

## 2. `02_data_pipelines_and_etl.ipynb`

- Load ROI config
- **Synthetic satellite-like channels** (NO2, CO, Aerosol Index) with heatmaps
- **Align to 64×64** (bicubic) and visualize
- **Normalize** (minmax) and compare before/after
- Stack days, **save/load** `.pt` tensor
- **Multi-day channel visualizations**

## 3. `03_model_and_predictions.ipynb`

- Load **FusionModel**, create dummy (or real) inputs
- **Forward pass** → PM2.5 map (B, 64, 64)
- **PM2.5 heatmap** over grid
- **Geographic overlay** (extent = ROI)
- Batch view for multiple samples

## Requirements

- Python 3.10+
- `matplotlib`, `numpy`, `torch`, `scipy`, `pyyaml`
- From project root: `pip install -r requirements.txt`
