# PathSnap Implementation Progress

## Status Overview
- **Current Stage**: ALL MILESTONES (1–5) COMPLETED & VERIFIED
- **Workspace**: `/Users/sushirc/Downloads/ML_HACKATHON `

## Completed Tasks
- [x] **Milestone 1 — EDA & First-Phase Pipeline**:
  - Audited all 80 training trips (`208,417` GPS observations) and 10 student-testing trips (`19,160` observations); `data/evaluator_test/` configured separately for runtime evaluation.
  - Built per-trip WGS84 UTM projection (auto-handling 25 UTM zones across N & S hemispheres), CSR directed road graph + LRU-cached bounded Dijkstra, vectorized Shapely 2.0 `STRtree`, banded monotonic DP label alignment, and 36 strictly prediction-time features.
  - Saved `experiments/dataset_inventory.csv`, `experiments/eda_summary.json`, `experiments/label_alignment_summary.csv`, EDA figures, and interactive Folium map `reports/maps/eda_trip_00000001.html`.
- [x] **Milestone 2 — 6-Model Benchmark & Bayesian Optimization**:
  - Benchmarked 6 algorithm families (`Logistic Regression`, `Distance-Weighted KNN`, `Decision Tree`, `Random Forest`, `HistGradientBoosting`, `MLP Neural Network`) via 5-fold `GroupKFold` CV + 10-trip student-test (`experiments/model_comparison.csv`, `experiments/model_comparison_details.json`).
  - Selected **HistGradientBoostingClassifier** (highest CV Macro F1 `0.8933 ± 0.0218`, highest STest Top-1 `83.17%`, lowest Log Loss `0.0689`).
  - Executed 15-trial Optuna TPE Bayesian optimization (`src/models/optimize.py`), improving STest Macro F1 to `0.9059` and Log Loss to `0.0687`. Saved `artifacts/models/final_model.joblib` (2.0 MB), `artifacts/preprocessors/feature_config.json`, `artifacts/configurations/best_hyperparameters.json`, and `experiments/optimization_trials.csv`.
- [x] **Milestone 3 — HMM & Viterbi Route Reconstruction**:
  - Implemented numerically stable log-space Viterbi decoder (`src/decoder/`) with road-network shortest-path transitions, GPS-gap relaxation, disconnected-graph segmentation recovery, and intermediate arc bridging.
  - Tuned HMM parameters on 8 training trips (`artifacts/configurations/hmm_config.json`) and evaluated across all 10 student-testing trips (`experiments/hmm_vs_classifier.csv`, `experiments/hmm_summary.json`):
    - **Weighted Top-1 Accuracy**: `83.08%` → **`88.55%`** (+5.47 pp; improved on **10/10 trips**)
    - **Sequence Length-Weighted Route F1**: `92.06%` → **`98.17%`**
    - **Reconstructed Direct Continuity**: `78.34%` → **`99.99%`** (100% on 9/10 trips; disconnected transitions cut from `330` → `3`)
  - Generated interactive Folium comparison HTML maps (`reports/maps/student_test_*_comparison.html`) and static exports (`reports/figures/map_student_test_*.png`).
- [x] **Milestone 4 — FastAPI Backend, React/Leaflet UI & Two-Laptop Deployment**:
  - Built FastAPI backend (`src/api/`, `0.0.0.0:8000`) with strict prediction-time vs. server-side evaluation separation (`/api/health`, `/api/trips`, `/api/trips/{trip_id}/predict`, `/api/trips/{trip_id}/map`).
  - Verified all 10 student-testing trips live through FastAPI (`scripts/verify_api_all_trips.py`), saving `experiments/logs/api_verification.log` and `reports/figures/api_logs_screenshot.png`.
  - Built React + Vite + TypeScript + Leaflet Evaluator Dashboard (`frontend/`) with `VITE_API_BASE_URL` configuration, verified clean TypeScript + Vite production build (`npm --prefix frontend run build`).
- [x] **Milestone 5 — Submission Deliverables & Reserved Evaluator-Test Evaluation**:
  - Extracted `evaluator_test.zip` (10 reserved trips: `00000015`..`00000094`) into `data/evaluator_test/` and verified automatic activation in FastAPI (`active_partition='evaluator_test'`, `is_development_fallback=False`, zero `.route` leakage).
  - Ran strictly frozen inference (zero retraining/tuning; all `train` and `student_test` results preserved) followed by post-prediction server-side evaluation (`scripts/run_evaluator_evaluation.py`):
    - **Weighted Top-1 Accuracy**: `76.94%` (Clf-Only) → **`89.13%`** (HMM, +12.18 pp; improved 10/10 trips)
    - **Mean Trip Top-1 Accuracy**: `84.31%` → **`90.24%`**
    - **Sequence Length-Weighted Route F1**: `88.54%` → **`98.20%`** (Trip `00000067` = `100.00%`)
    - **Reconstructed Direct Continuity**: `79.90%` → **`99.96%`** (100.00% on 9/10 trips)
  - Generated consolidated technical report PDF (`reports/PathSnap_Technical_Report.pdf`, Sections 2.2–8) and standalone final evaluation report (`reports/PathSnap_Final_Evaluation_Report.pdf`).
  - Wrote complete `README.md`, `.env.example`, `.gitignore`, launch scripts (`scripts/run_backend.sh`, `scripts/run_frontend.sh`), and `notebooks/exploratory_analysis.ipynb`.

## Latest Successful Tests
- `.venv/bin/pytest -v`: **7/7 passed** (`tests/test_core_pipeline.py` + `tests/test_api.py`).
- `.venv/bin/python scripts/verify_api_all_trips.py`: **10/10 student_test trips HTTP 200**, zero ground-truth leakage verified.
- `npm --prefix frontend run build`: **TypeScript + Vite build passed cleanly**.

## Optional Final Capture Before Submission
- When you launch the UI in your browser on Laptop 2 (`./scripts/run_backend.sh` + `./scripts/run_frontend.sh`), you can optionally save a browser screenshot to `reports/figures/ui_dashboard_screenshot.png` and a Folium HTML browser view to `reports/figures/folium_map_screenshot.png`, then re-run `.venv/bin/python scripts/generate_report.py` to automatically embed those two live browser screenshots into Section 6 & 7 of `reports/PathSnap_Technical_Report.pdf`.

## How to Run
```bash
# Laptop 1 (Backend on 0.0.0.0:8000)
./scripts/run_backend.sh

# Laptop 2 (Frontend on 0.0.0.0:5173, pointing to Laptop 1 IP)
VITE_API_BASE_URL=http://<LAPTOP1_IP>:8000 ./scripts/run_frontend.sh

# Run full test suite
.venv/bin/pytest -v
```
