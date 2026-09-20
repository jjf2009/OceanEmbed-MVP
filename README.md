# OceanEmbed — MVP

Reconstructs **subsurface ocean temperature** from **surface satellite fields**, at **0.25° daily**
resolution on **15 standard depth levels (0–1000 m)** — the Smart India Hackathon problem statement
surveyed in `OceanEmbed_Research_Synopsis.pdf`.

This is a deliberately thin end-to-end slice: a small Arabian Sea box, two years, a small model.
The point is that the whole loop closes — data → embedding → profile → metrics → clickable UI.

## Architecture

Following **TS-Cast** (the closest prior work in the synopsis): surface variables are compressed
into a compact **64-dimensional embedding** per pixel, then decoded into a 15-level temperature
profile with a **second head predicting per-level uncertainty**.

```
sst, ssh, sss, sin/cos(day-of-year), lat, lon
        │
     encoder MLP  ──────►  64-d embedding
        │
     decoder MLP  ──────►  15 × mu  +  15 × log-variance
```

Trained with Gaussian NLL, so the model can say *where* it is unsure — near the thermocline it
should be, at 1000 m it should not.

Everything implements `models/base.py::ReconstructionModel`, so a ConvLSTM / U-Net / GNN backbone
drops in without touching training, evaluation, inference or the API.

## Region and data

| | |
|---|---|
| Box | 10°N–20°N, 60°E–75°E (Arabian Sea) |
| Period | 2021-01-01 → 2022-12-31, daily |
| Grid | 0.25° |
| Levels | 0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 400, 500, 700, 1000 m |
| Source | Copernicus GLORYS `cmems_mod_glo_phy_my_0.083deg_P1D-m` |

GLORYS supplies both the surface inputs and the subsurface truth, so the MVP needs one product and
no ARGO co-location.

**Splits are time-blocked**, not random: 2021 trains, H1 2022 validates, H2 2022 tests. A random
pixel split would put a pixel's own neighbours — and its next day — in the test set, and the
reported skill would be fiction. Normalisation statistics come from the train block alone.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Data --------------------------------------------------------------
cp .env.example .env            # add your free CMEMS credentials
python -m oceanembed.data.download --config configs/mvp.yaml
# no credentials yet? everything below still runs:
python -m oceanembed.data.download --config configs/mvp.yaml --synthetic

# 2. Pipeline ----------------------------------------------------------
python -m oceanembed.data.preprocess --config configs/mvp.yaml
python -m oceanembed.train          --config configs/mvp.yaml
python -m oceanembed.evaluate       --config configs/mvp.yaml

# 3. Demo --------------------------------------------------------------
uvicorn oceanembed.api:app --reload      # http://127.0.0.1:8000
```

`--synthetic` writes a physically-shaped fake cube (exponential thermocline whose scale depth is
driven by SSH, so SST alone is not enough to solve it). The UI shows a **SYNTHETIC DATA** banner
whenever it is serving that cube — numbers from it are a pipeline check, never a result.

## Evaluation

`evaluate.py` reports per-depth RMSE / MAE / bias / R² on the held-out test block against two
baselines:

- **climatology** — the train-block mean profile for that calendar month
- **linear** — least squares on the same 7 features

A model that cannot beat monthly climatology has learnt nothing worth claiming, so the summary
prints how many of the 15 levels it wins. Output lands in `reports/metrics.json` and
`reports/rmse_by_depth.png`, and the UI renders the table underneath the map.

## API

| Route | Returns |
|---|---|
| `GET /health` | liveness |
| `GET /meta` | region, depth levels, date range, synthetic flag, latest metrics |
| `GET /predict/profile?lat&lon&date` | depths, `mu`, `sigma`, GLORYS `truth` |
| `GET /predict/field?date&depth` | 2-D predicted grid for the map overlay |
| `GET /` | the demo UI |

## UI

Leaflet map of the box coloured by the predicted field at the selected depth and date. Click any
cell and Plotly draws the predicted profile with a ±1σ band against the GLORYS truth curve. A date
slider moves both.

## Tests

```bash
pytest
```

Covers the processed-cube shape, feature/target shapes, the normalisation roundtrip, that the three
time splits do not overlap, that normalisation statistics were not leaked from val/test, model
shapes, that the NLL actually penalises confident errors, and a train → evaluate → API smoke run on
a miniature synthetic cube.

## Not in the MVP

Named deliberately, and mapped onto the synopsis's own reading-priority list:

- **ConvLSTM / ViT / GNN backbones** — Zhang 2024, Convformer, STGAT. The `ReconstructionModel`
  interface exists for exactly this.
- **Physics-informed loss** — physics-guided ConvLSTM, OG-PINN. Relevant to Arabian Sea upwelling.
- **Cloud-gap filling** — MAESSTRO's masked autoencoder, for real satellite SST rather than reanalysis.
- **Input latency handling** — Attention 3D-U-Net++: real SSS lags 3–7 days, wind 1–3 days.
- **Independent validation** — EN4 / IAP / Ishii / ARGO, rather than scoring GLORYS against itself.
- **Full North Indian Ocean box**, salinity output, and the adaptive spatiotemporal clustering trick
  (reported 12–27% RMSE gains, cheap to add).
# SIH26066_Protoype_Demo
