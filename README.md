# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap is an end-to-end distributed machine learning and geospatial route reconstruction system that matches noisy 1 Hz GPS trajectories onto directed road networks using spatial indexing, 36 domain-engineered prediction-time features, six benchmarked ML model families, Bayesian hyperparameter optimization (Optuna TPE), and a road-network-aware Hidden Markov Model (HMM) with numerically stable log-space Viterbi decoding.

The system is engineered for **two-laptop distributed deployment**:
- **Laptop 1 (Backend)**: FastAPI + Uvicorn server hosting the trained classifier, spatial index, directed road-network CSR graph, HMM/Viterbi decoder, and isolated server-side evaluation.
- **Laptop 2 (Frontend)**: React + Vite + TypeScript + Leaflet Evaluator Dashboard communicating strictly over HTTP via `VITE_API_BASE_URL`.

---

## Repository Structure

```text
PathSnap/
├── src/
│   ├── config.py                    # Configurable dataset, artifact, and experiment paths
│   ├── data/
│   │   ├── loader.py                # Strict separation: load_trip_inputs() never reads .route
│   │   ├── validation.py            # Coordinate, timestamp, graph, and route integrity audit
│   │   └── preprocessing.py         # Per-trip WGS84 UTM projection & GPS kinematics
│   ├── geo/
│   │   ├── projection.py            # Automatic per-trip WGS84 UTM EPSG selector (N & S hemispheres)
│   │   ├── geometry.py              # Vectorized point-to-segment projection & compass bearings
│   │   ├── road_graph.py            # CSR directed adjacency graph + cached bounded Dijkstra
│   │   └── spatial_index.py         # Vectorized Shapely 2.0 STRtree candidate index
│   ├── features/
│   │   ├── candidate_generation.py  # Configurable radius/top-K candidate retrieval
│   │   ├── feature_extraction.py    # 36 prediction-time features (zero leakage)
│   │   └── label_alignment.py       # Banded monotonic DP GPS-to-route alignment (offline only)
│   ├── models/
│   │   ├── dataset_cache.py         # Feature & label cache builder
│   │   ├── registry.py              # 6 ML model families (LR, KNN, DT, RF, HGB, MLP)
│   │   ├── evaluate.py              # Candidate, observation Top-1, and route continuity metrics
│   │   ├── compare.py               # 5-fold GroupKFold CV + student-test 6-model benchmark
│   │   ├── optimize.py              # Optuna TPE Bayesian hyperparameter optimization
│   │   └── train.py                 # End-to-end reproducible training orchestrator
│   ├── decoder/
│   │   ├── transitions.py           # Road-network shortest-path transition model & gap handling
│   │   ├── viterbi.py               # Log-space Viterbi decoder with segmentation recovery
│   │   └── hmm.py                   # High-level PathSnapHMM wrapper & serializable config
│   ├── evaluation/
│   │   ├── metrics.py               # Accuracy, Macro F1, Precision, Recall, Log Loss, Top-1
│   │   └── route_metrics.py         # Topological continuity & sequence-level route F1
│   ├── visualization/
│   │   ├── folium_maps.py           # Interactive Folium map builder (EDA & evaluator-safe)
│   │   └── plots.py                 # High-DPI matplotlib route & metric figures
│   └── api/
│       ├── main.py                  # FastAPI app, CORS, request logging middleware
│       ├── routes.py                # /api/health, /api/trips, /predict, /map endpoints
│       └── services.py              # Isolated inference service + post-prediction server eval
├── frontend/                        # React + Vite + TypeScript + Leaflet Evaluator Dashboard
├── artifacts/                       # Saved trained model, feature config, HMM & Optuna configs
├── experiments/                     # Real CSV/JSON experiment logs (EDA, models, Optuna, HMM)
├── reports/
│   ├── figures/                     # Actual generated plots, map exports, UI & API screenshots
│   ├── maps/                        # Interactive Folium HTML maps
│   └── PathSnap_Technical_Report.pdf
├── notebooks/
│   └── exploratory_analysis.ipynb
├── scripts/                         # Reproducible EDA, cache, HMM, report, and launch scripts
├── tests/                           # Pytest unit, pipeline, API, and ground-truth isolation tests
├── requirements.txt
└── .env.example
```

---

## Setup & Installation

### 1. Backend & ML Environment (Laptop 1)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Place the raw dataset partitions in the configured directories (or override paths in `.env`):
- `data/raw/train/` — 80 training trips
- `data/raw/student_test/` — 10 student-testing trips
- `data/evaluator_test/` — 10 reserved evaluator-testing trips (when supplied at evaluation time)

### 2. Frontend Environment (Laptop 2)
```bash
cd frontend
npm install
cp .env.example .env
```

---

## Reproducing All Experiments

Run the complete experimental pipeline from scratch:

```bash
# 1. Dataset audit, inventory CSV, EDA plots, and Folium EDA map
.venv/bin/python scripts/run_eda.py

# 2. Build processed feature & monotonic DP label cache (80 train + 10 student_test)
.venv/bin/python scripts/build_feature_cache.py

# 3. Train & compare all 6 ML model families (5-fold GroupKFold CV + student_test)
.venv/bin/python src/models/compare.py

# 4. Run Bayesian hyperparameter optimization (Optuna TPE) and save final model artifact
.venv/bin/python src/models/optimize.py

# 5. Tune HMM/Viterbi and compare Classifier-Only vs. Classifier + HMM across all 10 student_test trips
.venv/bin/python scripts/run_hmm_experiments.py

# 6. Run automated test suite (core geospatial, Viterbi, API & ground-truth isolation)
.venv/bin/pytest -v

# 7. Regenerate consolidated technical report PDF
.venv/bin/python scripts/generate_report.py
```

---

## Two-Laptop Distributed Deployment Guide

### Step 1: Connect Both Laptops to the Same Network
Connect **Laptop 1** (Backend/ML) and **Laptop 2** (Evaluator UI) to the same Wi-Fi network or mobile hotspot.

### Step 2: Find Laptop 1's Local LAN IP Address
- **macOS**: `ipconfig getifaddr en0` (or `ifconfig | grep "inet " | grep -v 127.0.0.1`)
- **Linux**: `hostname -I`
- **Windows (CMD)**: `ipconfig` (look for `IPv4 Address` under your Wi-Fi adapter, e.g., `192.168.1.42`)

### Step 3: Load Reserved Evaluation Trips & Start Backend on Laptop 1
When the evaluator supplies the 10 reserved trips, place them inside `data/evaluator_test/` (each trip in its own `<trip_id>/` folder with `.track`, `.nodes`, `.arcs`, and server-side `.route` if supplied). The backend automatically detects and serves `data/evaluator_test/` when populated, and clearly labels `student_test` as development-only otherwise.

Start FastAPI + Uvicorn listening on all interfaces (`0.0.0.0:8000`) with live access logs visible:
```bash
./scripts/run_backend.sh
# Equivalent direct command:
.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --access-log --log-level info
```

Keep this terminal visible during evaluation so the evaluator can observe incoming HTTP requests (`GET /api/trips`, `POST /api/trips/{trip_id}/predict`, `GET /api/trips/{trip_id}/map`) and prediction timing logs in real time.

### Step 4: Configure & Start Frontend on Laptop 2
Edit `frontend/.env` on Laptop 2 to point to Laptop 1's LAN IP:
```env
VITE_API_BASE_URL=http://192.168.1.42:8000
```

Start the Evaluator Dashboard:
```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```
Open `http://localhost:5173` in the browser on Laptop 2.

### Step 5: Verify Cross-Laptop Connectivity
From Laptop 2's terminal:
```bash
# Verify health and loaded model
curl http://192.168.1.42:8000/api/health

# Verify dynamically discovered evaluation trips
curl http://192.168.1.42:8000/api/trips

# Trigger a live prediction request (check Laptop 1's terminal for live logs)
curl -X POST http://192.168.1.42:8000/api/trips/<TRIP_ID>/predict
```

---

## Ground-Truth Separation & Evaluation Integrity

- Prediction-time execution (`run_prediction_only`) reads **only** `.track`, `.nodes`, `.arcs`, and saved artifacts (`artifacts/models/final_model.joblib`, `artifacts/preprocessors/feature_config.json`, `artifacts/configurations/hmm_config.json`).
- Server-side evaluation (`evaluate_server_side`) runs strictly **after** prediction completes, reads `.route` exclusively on Laptop 1, and returns only aggregate scalar metrics (returning `null` / `Unavailable` whenever no `.route` exists).
- Neither `.route` arc sequences nor reference-route geometries are ever sent to the model, API response payload, evaluator Folium map, or frontend dashboard.

---

## Team Members & Contributions

All four team members contributed equally (**25% each**) across distinct technical workstreams of the PathSnap system.

| # | Name | SRN | Role | Contribution |
|---|---|---|---|---|
| 1 | **Sushir C** | `PES1UG24AM297` | Data & Geospatial Foundation | 25% |
| 2 | **T Surya** | `PES1UG24AM300` | Features & Machine Learning Model | 25% |
| 3 | **T B Akash** | `PES1UG24AM299` | HMM, Evaluation & Quality | 25% |
| 4 | **Swastik Pamadi** | `PES1UG24AM298` | Application, API & Release Integration | 25% |

### 1. Sushir C
- **SRN:** `PES1UG24AM297`
- **Role:** Data & Geospatial Foundation
- **Contribution:** 25%
- **Responsibilities:**
  - Dataset validation and exploratory data analysis (EDA).
  - Coordinate projection and geospatial preprocessing.
  - Road graph construction.
  - Spatial indexing and candidate road search.
  - Dataset audit and visualisation of GPS observations with nearby road candidates.
  - Testing graph construction and spatial search.

### 2. T Surya
- **SRN:** `PES1UG24AM300`
- **Role:** Features & Machine Learning Model
- **Contribution:** 25%
- **Responsibilities:**
  - Route-label alignment.
  - Prediction-time feature engineering.
  - Training and evaluation of machine learning models.
  - Model comparison and Bayesian hyperparameter optimisation.
  - Saved model artifacts and experiment results.
  - Reproducible training pipeline and performance metrics.

### 3. T B Akash
- **SRN:** `PES1UG24AM299`
- **Role:** HMM, Evaluation & Quality
- **Contribution:** 25%
- **Responsibilities:**
  - Road-aware transition scoring.
  - Hidden Markov Model (HMM) implementation.
  - Viterbi route decoding.
  - Route evaluation metrics and continuity analysis.
  - Held-out evaluator testing.
  - Ground-truth isolation and verification that reference routes do not leak into predictions.
  - Classifier-only versus HMM performance comparisons.

### 4. Swastik Pamadi
- **SRN:** `PES1UG24AM298`
- **Role:** Application, API & Release Integration
- **Contribution:** 25%
- **Responsibilities:**
  - FastAPI backend development and endpoints.
  - React/Leaflet evaluator dashboard.
  - Two-laptop deployment and integration.
  - Frontend/backend communication.
  - Live prediction demonstration.
  - Launch scripts, README documentation and deployment instructions.
