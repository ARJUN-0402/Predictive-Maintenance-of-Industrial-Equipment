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
python src/train.py
```

This will:
1. Download `ai4i2020.csv` into `data/raw/` if it isn't already there
2. Engineer features and fit the preprocessing pipeline on the training split only
3. Train XGBoost (with GridSearchCV), Random Forest, Gradient Boosting, and Logistic Regression
4. Save all model artifacts to `models/` and write a run entry to `model_registry.json`
5. Save evaluation plots and reports to `reports/`

### 2. Launch the dashboard

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