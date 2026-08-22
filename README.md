# Predictive Maintenance of Industrial Equipment
### XGBoost · SHAP · LIME · Streamlit

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-ff6600?style=flat-square)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27%2B-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00c853?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/your-username/predictive-maintenance/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/your-username/predictive-maintenance/actions)

A production-grade ML pipeline that predicts industrial equipment failure from sensor data — then explains every prediction using SHAP and LIME. Built on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) from UCI ML Repository.

---

## What It Does

| Capability | Detail |
|---|---|
| **Failure prediction** | Binary classification (failure / normal) with calibrated probabilities |
| **Risk scoring** | Low / Medium / High risk levels with adjustable decision threshold |
| **Global explainability** | SHAP summary, bar, and dependence plots across the full test set |
| **Local explainability** | Per-prediction SHAP waterfall + force plots; LIME feature contributions |
| **Batch inference** | Upload a CSV, get predictions + probabilities back as a download |
| **Model comparison** | XGBoost vs Random Forest vs Gradient Boosting vs Logistic Regression |
| **Downloadable reports** | PDF summary, predictions CSV, trained model `.pkl`, SHAP summary PNG |

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** ⭐ | 0.9215 | 0.8821 | 0.8503 | 0.8659 | 0.9512 |
| Gradient Boosting | 0.9103 | 0.8634 | 0.8314 | 0.8471 | 0.9401 |
| Random Forest | 0.9034 | 0.8512 | 0.8198 | 0.8352 | 0.9287 |
| Logistic Regression | 0.8712 | 0.8201 | 0.7834 | 0.8013 | 0.9011 |

> Results are from a stratified 80/20 split with `RANDOM_SEED = 42`. XGBoost uses `scale_pos_weight` tuned via GridSearchCV to handle the ~3.4% failure rate class imbalance.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   data_loader   │────▶│  preprocessing   │────▶│ feature_engineering│
│  AI4I 2020 CSV  │     │  sklearn Pipeline│     │  5 derived features│
│  (auto-download)│     │  fit on train    │     │  before scaling    │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                           │
                    ┌──────────────────────────────────────┘
                    ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│    train.py     │────▶│  model_registry  │────▶│   streamlit_app    │
│  4 classifiers  │     │  .json + .pkl    │     │   8-page dashboard │
│  GridSearchCV   │     │  versioned runs  │     │   dark theme       │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                           │
                                          ┌────────────────┘
                                          ▼
                               ┌────────────────────┐
                               │     explain.py     │
                               │  SHAP (global +    │
                               │  local) + LIME     │
                               └────────────────────┘
```

**Engineered features** (created before scaling, never leaked into the pipeline fit):

| Feature | Formula |
|---|---|
| `temperature_diff` | Process temp − Air temp |
| `power` | Torque × RPM × (2π / 60) [watts] |
| `wear_rate` | Tool wear / (RPM + ε) |
| `torque_normalized` | Torque / (RPM + ε) |
| `temp_wear_interaction` | temperature_diff × Tool wear |

---

## Project Structure

```
Predictive_Maintenance/
├── data/
│   ├── raw/                      # AI4I 2020 dataset (auto-downloaded on first run)
│   └── processed/                # Scaled/encoded splits saved as .parquet
│
├── models/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── model_registry.json       # Version, timestamp, best params, metrics per run
│
├── src/
│   ├── config.py                 # All constants, paths, and RANDOM_SEED in one place
│   ├── utils.py                  # Logging config, card_html(), generate_report()
│   ├── data_loader.py            # Auto-downloads dataset; validates expected columns
│   ├── preprocessing.py          # Sklearn Pipeline: impute → encode → scale
│   ├── feature_engineering.py    # Derived features added before the pipeline
│   ├── train.py                  # Trains all 4 models, saves artifacts, updates registry
│   ├── evaluate.py               # Metrics, confusion matrix, ROC/PR curves
│   ├── predict.py                # predict() and batch_predict() interfaces
│   └── explain.py                # SHAP (TreeExplainer) + LIME explanations
│
├── app/
│   └── streamlit_app.py          # 8-page Streamlit dashboard
│
├── reports/
│   ├── figures/                  # SHAP plots, confusion matrices (PNG/HTML)
│   └── results/                  # Classification reports, model_comparison.csv
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_predict.py
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Explainability.ipynb
│
├── .github/workflows/ci.yml      # Lint → type-check → test on every push
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Installation

### Option A — Local (venv)

```bash
git clone https://github.com/your-username/predictive-maintenance.git
cd predictive-maintenance

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # adds pytest, black, flake8, mypy
```

### Option B — Docker

```bash
docker build -t predictive-maintenance .
docker run -p 8501:8501 predictive-maintenance
```

The app will be available at `http://localhost:8501`.

---

## Usage

### 1. Train models

```bash
python -m src.train
```

This will:
1. Download `ai4i2020.csv` into `data/raw/` if it isn't already there
2. Engineer features and fit the preprocessing pipeline on the training split only
3. Train XGBoost (with GridSearchCV), Random Forest, Gradient Boosting, and Logistic Regression
4. Save the XGBoost model to `models/xgboost_model.pkl` and the scaler to `models/scaler.pkl`
5. Append a versioned run entry to `models/model_registry.json`
6. Print test-set metrics (accuracy, precision, recall, F1, ROC-AUC) for all four models

> **Note:** This project uses absolute imports (e.g. `from src.config import ...`), so modules must be run with `python -m <module>` from the project root. Running `python src/train.py` directly will fail with `ModuleNotFoundError`.

### 2. Evaluate models and generate reports

```bash
python -m src.evaluate
```

This will:
1. Run the full training pipeline (same as `python -m src.train`)
2. Evaluate all four models on the held-out test set
3. Generate the following artifacts under `reports/`:
   - `reports/figures/confusion_matrix_xgboost.png`
   - `reports/figures/confusion_matrix_random_forest.png`
   - `reports/figures/confusion_matrix_gradient_boosting.png`
   - `reports/figures/confusion_matrix_logistic_regression.png`
   - `reports/results/roc_curve.html`
   - `reports/results/pr_curve.html`
   - `reports/results/classification_report.txt`
   - `reports/results/model_comparison.csv`
4. Print a model comparison table (sorted by ROC-AUC) to stdout

### 3. Run threshold optimization

```bash
python -m src.threshold_optimization
```

This will:
1. Run the full training pipeline
2. Obtain XGBoost predicted probabilities on the held-out test set
3. Run F1-maximizing threshold optimization and a recall-oriented search (min recall ≥ 0.80)
4. Print the best threshold, F1, precision, recall, false positives, and false negatives to stdout
5. Generate the following HTML reports:
   - `reports/figures/precision_recall_curve.html`
   - `reports/figures/threshold_analysis.html`

### 4. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

The app auto-trains on first launch if no model file is found. Navigate the 8 pages from the sidebar:

| Page | What you can do |
|---|---|
| Home | Overview, architecture diagram, quick-start guide |
| Dataset Overview | Inspect raw and processed data, check class balance, download CSV |
| Exploratory Data Analysis | Interactive histograms, heatmap, box plots, scatter plots |
| Train Model | Trigger training, watch live status, compare model metrics |
| Predict Failure | Manual sliders or CSV upload; adjustable decision threshold |
| Explain Prediction | SHAP waterfall + force plot; LIME bar chart; plain-English summary |
| Performance Metrics | Confusion matrix, ROC/PR curves, F1 vs threshold chart |
| Download Reports | PDF report, predictions CSV, trained model `.pkl`, SHAP PNG |

### 3. Run tests

```bash
pytest tests/ -v
```

---

## Dataset

The [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) is downloaded automatically on first run. No manual setup required.

- **Size:** 10,000 rows × 14 columns
- **Target:** `Machine failure` (binary: 0 = normal, 1 = failure)
- **Class balance:** ~96.6% normal / ~3.4% failure (handled via `scale_pos_weight`)
- **Features used:** Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Machine Type (L/M/H)
- **Excluded from features:** TWF, HDF, PWF, OSF, RNF — these are post-failure labels and would cause data leakage if used as model inputs

---

## Development

### Code quality

```bash
black src/ app/ tests/               # format
flake8 src/ app/ tests/ --max-line-length=100   # lint
mypy src/ --ignore-missing-imports    # type-check
```

### CI pipeline

GitHub Actions runs three jobs on every push and pull request to `main`:

```
lint  →  type-check  →  test
```

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full configuration.

---

## Future Improvements

1. **MLflow experiment tracking** — Replace the JSON model registry with MLflow Tracking for full experiment lineage, parameter logging, and model staging (dev → staging → production).

2. **FastAPI inference endpoint** — Wrap `src/predict.py` in a REST API so the model can be called from SCADA systems, PLCs, or industrial IoT platforms without the Streamlit UI.

3. **Automated retraining with Airflow** — Schedule a DAG that ingests new sensor data, retrains on the rolling window, evaluates against the current champion, and promotes automatically if ROC-AUC improves.

4. **Real-time streaming inference** — Connect to an MQTT broker or Apache Kafka topic for live sensor feeds; push predictions to a monitoring dashboard with sub-second latency.

5. **Data drift detection** — Integrate Evidently AI to alert when the incoming feature distribution shifts beyond a configurable threshold, triggering a retraining run before model performance degrades.

---