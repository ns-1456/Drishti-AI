# Vayu-Mamba (Delhi NCR Edition)

Hyper-local pollution attribution for Delhi NCR: Delhi, Noida, Gurugram, Ghaziabad, Faridabad.  
Optimized for **high-precision** and **low-latency** on Google Colab Pro. Timeline: 4–6 weeks.

## Goal

Fuse Sentinel-5P satellite data (NO2, CO, Aerosol Index), social media (CLIP + Landmark NER), and CPCB ground truth via a Vision Mamba + Triton fusion engine. Demo: Streamlit app with stubble-burning evaluation (Nov 2024).

## Structure

```
Drishti-AI/
├── config/           # ROI, landmarks
├── data/             # satellite .pt, CPCB, social CSV (gitignored)
├── notebooks/        # Jupyter: overview, data pipelines, model + visualizations
├── src/
│   ├── pipeline/     # Phase 1: GEE → 64×64 grid → .pt
│   ├── social/       # Phase 2: Twitter + NER + CLIP → CSV
│   ├── model/        # Phase 3: Vim-Base + Point-Net + Triton fusion
│   └── app/          # Phase 4: Streamlit demo
├── scripts/          # One-off scripts (e.g. 7-day NCR download)
└── tests/
```

## Setup

1. Copy `.env.example` to `.env` and fill in GEE credentials and Twitter/X API keys.
2. `pip install -r requirements.txt`
3. Authenticate GEE: `earthengine authenticate` (once).

## Quick start

- **Phase 1:** `python scripts/download_ncr_7d.py` — sanity check GEE (7 days NO2, CO, Aerosol).
- **Satellite baseline (30 days):** `python scripts/download_ncr_30d.py` — save `data/satellite/ncr_30d.pt` for Vision Mamba training.
- **Offline ETL (batch):** `python scripts/run_offline_pipeline.py --days 30` — historical window; run on a schedule to refresh training data.
- **Online ETL (inference):** `python scripts/run_online_pipeline.py` — latest day(s) for inference or CL update.
- **Phase 4:** `streamlit run src/app/streamlit_demo.py` — run the demo app (after model + data exist).
- **Notebooks:** Open `notebooks/01_project_overview_and_architecture.ipynb`, `02_data_pipelines_and_etl.ipynb`, `03_model_and_predictions.ipynb` for walkthroughs and visualizations (ROI map, pipeline flow, channel heatmaps, PM2.5 prediction maps).

## Data pipelines (ETL)

Two pipelines for feeding new data into the model on a schedule:

- **Offline pipeline:** Batch historical satellite (and future: social, CPCB, meteo, fire). Output: training dataset (e.g. `.pt`). Run weekly/monthly for full retrain.
- **Online pipeline:** Latest N days for inference and optional continual-learning updates. Run daily or more frequently.

Pipeline code lives in `src/pipeline/` and is runnable via `scripts/run_offline_pipeline.py` and `scripts/run_online_pipeline.py`. Use this in a separate ETL/data-engineering project by importing from `src.pipeline` or running these scripts.

## Branches

- **main** — full working project (model + pipelines + app).
- **data-pipelines** — same repo; use for ETL/data-engineering focus (pipelines in `src/pipeline/`, `scripts/run_*_pipeline.py`).
- **model** — same repo; use for AI model focus (`src/model/`, FusionModel, Vim, Point-Net, SparseFusion).

## License

MIT (or your choice).
