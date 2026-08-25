# Predictive Maintenance of Industrial Equipment
### XGBoost · SHAP · LIME · Streamlit

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-ff6600?style=flat-square)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27%2B-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end ML system for predicting industrial equipment failure from sensor data. Built on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) from UCI ML Repository. Compares four classifiers, optimizes decision thresholds, and explains every prediction with SHAP and LIME. Interactive results are served through an 8-page Streamlit dashboard.

---

## What It Does

| Capability | Detail |
|---|---|
| **Failure prediction** | Binary classification (failure / normal) on held-out test data |
| **Probability-based risk scoring** | Model-predicted failure probabilities with adjustable decision threshold |
| **Adjustable threshold** | Separate F1-maximizing and recall-oriented threshold optimization |
| **SHAP explainability** | Global summary plots and per-prediction waterfall / force plots |
| **LIME explainability** | Local feature contributions for individual predictions |
| **Batch prediction** | CSV upload with bulk predictions and probabilities |
| **Model comparison** | Side-by-side metrics for XGBoost, Random Forest, Gradient Boosting, and Logistic Regression |
| **Downloadable reports** | Classification reports, ROC/PR curves, confusion matrices, and model artifacts |
| **Streamlit dashboard** | 8-page interactive UI for exploration, training, prediction, and explainability |

---

## Model Results

Stratified 80/20 split, `RANDOM_SEED = 42`. Class imbalance handled via `scale_pos_weight` (XGBoost) and `class_weight="balanced"` (other models).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.9840 | 0.7308 | 0.8382 | 0.7808 | 0.9833 |
| Random Forest | 0.9880 | 0.8548 | 0.7794 | 0.8154 | 0.9778 |
| Gradient Boosting | 0.9900 | 0.9138 | 0.7794 | 0.8413 | 0.9754 |
| Logistic Regression | 0.8730 | 0.1921 | 0.8529 | 0.3135 | 0.9384 |

**Interpretation**

- **XGBoost** achieves the highest ROC-AUC (0.9833), indicating the strongest overall separability between failure and normal classes.
- **Gradient Boosting** delivers the highest F1 (0.8413) and precision (0.9138), making it suitable when false alarms are expensive.
- **Logistic Regression** attains the highest recall (0.8529) but at the cost of very low precision (0.1921).
- **Model selection depends on the operational objective and the relative cost of false positives vs. false negatives.** No single model is universally optimal.

---

## Threshold Optimization

The decision threshold is optimized separately from model training using the XGBoost probability outputs on the test set. Two strategies are evaluated:

| Strategy | Description |
|---|---|
| **F1-maximizing** | Balances precision and recall for general-purpose deployment |
| **Recall-oriented** | Prioritizes high recall when missed failures are more costly than false alarms |

Threshold values are persisted in `models/model_registry.json` and loaded dynamically by the dashboard. No single threshold is claimed to be universally optimal for all industrial operations.

---

## Architecture

```
data loading
    → validation
    → feature engineering
    → preprocessing
    → model training
    → model registry
    → evaluation
    → threshold optimization
    → prediction
    → explainability
    → Streamlit dashboard
```

**Engineered features** (created before scaling, never leaked into pipeline fit):

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
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── validation.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── threshold_optimization.py
│   └── explain.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_evaluate.py
│   ├── test_feature_engineering.py
│   ├── test_predict.py
│   ├── test_preprocessing.py
│   ├── test_validation.py
│   └── test_cli.py
├── app/
│   └── streamlit_app.py
├── models/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── model_registry.json
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── figures/
│   └── results/
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Installation

### Option A — Local (venv)

```bash
git clone https://github.com/ARJUN-0402/Predictive-Maintenance-of-Industrial-Equipment.git
cd "Predictive Maintenance of Industrial Equipment"

python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # Unix / macOS

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

### 1. Launch the dashboard (recommended)

```bash
streamlit run streamlit_app.py
```

Opens the dashboard at `http://localhost:8501`. Pretrained model artifacts are loaded automatically when available. No training is required to make predictions.

### 2. Train models (optional)

```bash
python -m src.train
```

Downloads the dataset (if missing), engineers features, preprocesses the training split, trains all four classifiers, saves the best model and scaler to `models/`, updates `models/model_registry.json`, and prints test-set metrics.

### 3. Evaluate models and generate reports (optional)

```bash
python -m src.evaluate
```

Runs evaluation on the held-out test set and generates confusion matrices, ROC/PR curves, classification reports, and a model comparison CSV under `reports/`.

### 4. Run threshold optimization (optional)

```bash
python -m src.threshold_optimization
```

Searches for F1-maximizing and recall-oriented decision thresholds, prints optimal thresholds and metrics, saves HTML reports, and updates the model registry with the selected thresholds.

---

## Evaluation Artifacts

**models/**
- `models/xgboost_model.pkl`
- `models/scaler.pkl`
- `models/model_registry.json`

**data/processed/**
- `data/processed/train.parquet`
- `data/processed/test.parquet`

**reports/figures/**
- `reports/figures/confusion_matrix_xgboost.png`
- `reports/figures/confusion_matrix_random_forest.png`
- `reports/figures/confusion_matrix_gradient_boosting.png`
- `reports/figures/confusion_matrix_logistic_regression.png`
- `reports/figures/precision_recall_curve.html`
- `reports/figures/threshold_analysis.html`

**reports/results/**
- `reports/results/roc_curve.html`
- `reports/results/pr_curve.html`
- `reports/results/classification_report.txt`
- `reports/results/model_comparison.csv`

---

## Dataset

The [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) is downloaded automatically on first run.

- **Size:** 10,000 rows × 14 columns
- **Target:** `Machine failure` (binary: 0 = normal, 1 = failure)
- **Class balance:** ~96.6% normal / ~3.4% failure (handled via `scale_pos_weight` and `class_weight`)
- **Features used:** Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Machine Type (L/M/H)
- **Excluded from features:** TWF, HDF, PWF, OSF, RNF — these are post-failure labels and would cause data leakage if used as model inputs

---

## Model Registry

`models/model_registry.json` stores versioned model metadata for every training run. The registry uses a `shared` block at the top level for `dataset_info` and `feature_config`, so these configurations are stored once rather than duplicated across each version. Individual versions under `versions/` capture the model name, timestamp, best hyperparameters, and test-set metrics.

---

## Quality Gates

Verified current results:

```
pytest:           60 passed
mypy:             Success: no issues found in 12 source files
flake8:           0 errors
```

Commands:

```bash
python -m pytest tests/ --tb=short
python -m mypy src/ --ignore-missing-imports
python -m flake8 src/ app/ tests/ --max-line-length=100
```

---
## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app**, select the repository, and set the main file to `streamlit_app.py`.
4. Click **Deploy**.

The deployed app will be available at the URL provided by Streamlit Community Cloud.

---
## Streamlit Dashboard

The dashboard is organized around the primary prediction workflow:

| Section | Page |
|---|---|
| **Overview** | Home |
| **Data** | Dataset Overview, Exploratory Data Analysis |
| **Prediction** | Predict Failure, Explain Prediction |
| **Evaluation** | Performance Metrics |
| **Reports** | Download Reports |
| **Admin** | Train Model |

| Page | What you can do |
|---|---|
| Home | Landing page with metrics, architecture, and quick start |
| Dataset Overview | Inspect raw and processed data, check class balance, download CSV |
| Exploratory Data Analysis | Interactive histograms, heatmap, box plots, scatter plots |
| Predict Failure | Manual sliders or CSV upload; adjustable decision threshold |
| Explain Prediction | SHAP waterfall + force plot; LIME bar chart; plain-English summary |
| Performance Metrics | Confusion matrix, ROC/PR curves, F1 vs threshold chart |
| Download Reports | PDF report, predictions CSV, trained model `.pkl`, SHAP PNG |
| Train Model | Retrain models using the current dataset |

---

## Future Improvements

These are planned enhancements, not currently implemented:

1. **MLflow experiment tracking** — Replace the JSON model registry with MLflow Tracking for full experiment lineage, parameter logging, and model staging (dev → staging → production).

2. **FastAPI inference endpoint** — Wrap `src/predict.py` in a REST API so the model can be called from SCADA systems, PLCs, or industrial IoT platforms without the Streamlit UI.

3. **Automated retraining with Airflow** — Schedule a DAG that ingests new sensor data, retrains on a rolling window, evaluates against the current champion, and promotes automatically if ROC-AUC improves.

4. **Real-time streaming inference** — Connect to an MQTT broker or Apache Kafka topic for live sensor feeds; push predictions to a monitoring dashboard with sub-second latency.

5. **Data drift detection** — Integrate Evidently AI to alert when the incoming feature distribution shifts beyond a configurable threshold, triggering a retraining run before model performance degrades.
