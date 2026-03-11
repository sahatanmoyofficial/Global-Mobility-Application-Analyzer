# 🌍 Global Mobility Application Analyzer (GMAA)

> **Predicting US Visa Approval Outcomes with Machine Learning**
>
> An end-to-end MLOps system that takes 10 applicant & employer features and predicts whether a visa application will be **Certified ✅** or **Denied ❌** — with a production-ready FastAPI service, automated CI/CD to AWS, and a full 6-stage ML pipeline.

---

<div align="center">

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen?logo=mongodb)](https://www.mongodb.com/atlas)
[![AWS S3](https://img.shields.io/badge/AWS-S3%20%7C%20EC2%20%7C%20ECR-orange?logo=amazonaws)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📊 Project Slides

> **Want a visual overview first?** Browse the full presentation before diving into the code.

👉 **[View the Project Presentation on Google Slides](https://docs.google.com/presentation/d/1IBPzlvido2uV_umikUCwUiwCzNqzRQ-M/edit?usp=sharing&ouid=117459468470211543781&rtpof=true&sd=true)**

The deck covers: business problem → architecture → ML pipeline → model results → deployment → improvement roadmap — in 12 slides.

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Business Problem](#1-business-problem) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [High-Level Architecture](#4-high-level-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Data & Features](#6-data--features) |
| 7 | [ML Pipeline — Step by Step](#7-ml-pipeline--step-by-step) |
| 8 | [Model Performance](#8-model-performance) |
| 9 | [How to Replicate — Full Setup Guide](#9-how-to-replicate--full-setup-guide) |
| 10 | [Running the Application](#10-running-the-application) |
| 11 | [CI/CD & Cloud Deployment](#11-cicd--cloud-deployment) |
| 12 | [Business Applications & Other Industries](#12-business-applications--other-industries) |
| 13 | [How to Improve This Project](#13-how-to-improve-this-project) |
| 14 | [Troubleshooting](#14-troubleshooting) |
| 15 | [Glossary](#15-glossary) |

---

## 1. Business Problem

### What problem are we solving?

Every year, hundreds of thousands of employers file Labour Condition Applications (LCAs) with the US Department of Labor to sponsor foreign workers on H-1B and similar visas. The outcome — **Certified** or **Denied** — depends on a complex mix of factors that even experienced immigration consultants struggle to predict in advance.

This leads to real business pain:

- 💸 **Wasted legal fees** on applications that were never going to succeed
- ⏳ **Delayed workforce planning** when key hires fall through
- 🎯 **No actionable feedback** — rejected applicants don't know what to change
- ⚖️ **Inequitable outcomes** — similar profiles receive different decisions with no transparency

### What does GMAA answer?

> *"Given what we know about a job applicant and their sponsoring employer, what is the probability that the visa application will be Certified — and which factors matter most?"*

### Objectives

1. Build an ML model (target ROC-AUC ≥ 0.80) to predict visa certification
2. Expose predictions via a production-grade web API
3. Surface the key drivers of approval in human-readable form
4. Deliver a fully reproducible MLOps pipeline from raw data to live prediction
5. Deploy with full CI/CD — every code push triggers automated build & deploy

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Data Source** | MongoDB Atlas (`VISA_APPLICATION_DATA.visa_data`) |
| **Dataset** | 25,480 applications, 12 raw columns |
| **Target Variable** | `case_status` → Certified (0) or Denied (1) |
| **Train / Test Split** | 80 / 20 |
| **Class Balancing** | SMOTEENN (minority oversampling + ENN cleaning) |
| **Best Model** | KNN (k=3, distance weights) — CV Score **95.4%**, F1 **0.822** |
| **API Framework** | FastAPI + Uvicorn on port 8080 |
| **Model Storage** | AWS S3 bucket `visa2026` (eu-north-1) |
| **Deployment** | Docker → AWS ECR → AWS EC2 |
| **CI/CD** | GitHub Actions (push to `main` triggers full pipeline) |

---

## 3. Tech Stack

### Complete Technology Map

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Store** | MongoDB Atlas | Cloud-hosted NoSQL DB storing raw visa application records |
| **Language** | Python 3.8 | Core language for the entire ML pipeline and API |
| **Data Processing** | Pandas, NumPy | Data ingestion, splitting, feature engineering |
| **ML Framework** | Scikit-learn | Preprocessing pipelines, GridSearchCV, metrics |
| **Class Balancing** | imbalanced-learn | SMOTEENN for minority oversampling + ENN cleaning |
| **Model Selection** | neuro_mf | Config-driven multi-model factory with grid search |
| **Serialisation** | dill, pickle | Save/load model objects and preprocessor pipelines |
| **Drift Detection** | Evidently 0.2.8 | Statistical data drift between train and test sets |
| **Web API** | FastAPI | REST API with auto-generated OpenAPI docs |
| **ASGI Server** | Uvicorn | High-performance server running FastAPI |
| **Templating** | Jinja2 | Server-side HTML rendering for the prediction UI |
| **Frontend** | Bootstrap 5 | Responsive styling for the web form |
| **Cloud Storage** | AWS S3 | Model registry — stores `model.pkl` for production |
| **Cloud Compute** | AWS EC2 (Ubuntu) | Production server running the Docker container |
| **Container Registry** | AWS ECR | Stores Docker images pushed by GitHub Actions |
| **Containerisation** | Docker | Packages app + dependencies into a portable image |
| **CI/CD** | GitHub Actions | Automated build → push to ECR → deploy to EC2 |
| **Config** | YAML (PyYAML) | Schema config, model config, drift reports |
| **Logging** | Python logging | Timestamped log files per pipeline run in `/logs` |
| **EDA / Notebooks** | Jupyter, Seaborn, Plotly | Exploratory analysis and feature engineering |

---

## 4. High-Level Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│                                                                 │
│   [ Kaggle CSV ]  ──►  [ MongoDB Atlas ]                        │
│                         VISA_APPLICATION_DATA.visa_data         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE LAYER                           │
│                                                                 │
│  [Ingest]→[Validate]→[Transform]→[Train]→[Evaluate]→[Push]     │
│      │                                          │               │
│      ▼                                          ▼               │
│  artifact/<timestamp>/                    model.pkl → AWS S3   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SERVING LAYER                            │
│                                                                 │
│  [ FastAPI app.py ]  ◄── loads model.pkl from S3               │
│        │                                                        │
│   POST /  (predict)   GET /train  (retrain)                     │
│        │                                                        │
│  [ Browser / HTTP Client ]                                      │
│        ▲                                                        │
│  [ Docker Container ]  ◄──  [ AWS EC2 ]                        │
│                                  ▲                              │
│  [ GitHub Push ] → [ Actions ] → [ ECR ] → [ EC2 Deploy ]      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| # | Stage | What Happens |
|---|-------|-------------|
| 1 | **Data Loading** | Raw CSV uploaded to MongoDB Atlas; 25,480 records in collection `visa_data` |
| 2 | **Ingestion** | Pipeline pulls all records, saves `visa.csv`, splits 80/20 to `train.csv` / `test.csv` |
| 3 | **Validation** | Column count & type checks; Evidently drift report saved as YAML |
| 4 | **Transformation** | Feature engineering, encoding, scaling, SMOTEENN; output as `.npy` arrays |
| 5 | **Training** | `neuro_mf` runs GridSearchCV over KNN + RandomForest; best model saved |
| 6 | **Evaluation** | New model F1 vs. production F1; accepted if improvement > 0.02 |
| 7 | **Pushing** | Accepted `model.pkl` uploaded to S3 bucket `visa2026` |
| 8 | **Serving** | FastAPI loads model from S3; `POST /` returns `Certified` or `Denied` |

---

## 5. Repository Structure

```
Global-Mobility-Application-Analyzer/
├── visa/                           # Core Python package
│   ├── constants/__init__.py       # All global constants (paths, thresholds, names)
│   ├── entity/
│   │   ├── config_entity.py        # Dataclasses for component configs
│   │   ├── artifact_entity.py      # Dataclasses for component outputs
│   │   ├── estimator.py            # VisaModel wrapper + TargetValueMapping
│   │   └── s3_estimator.py         # VisaEstimator — load/save from S3
│   ├── components/
│   │   ├── data_ingestion.py       # Mongo → CSV → train/test split
│   │   ├── data_validation.py      # Schema checks + Evidently drift
│   │   ├── data_transformation.py  # Feature eng + encoding + SMOTEENN
│   │   ├── model_trainer.py        # GridSearchCV via neuro_mf
│   │   ├── model_evaluation.py     # Compare new vs. production model
│   │   └── model_pusher.py         # Upload accepted model to S3
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates all 6 components
│   │   └── prediction_pipeline.py  # Serves predictions at inference time
│   ├── configuration/
│   │   ├── mongo_db_connection.py  # MongoDB Atlas client (singleton)
│   │   └── aws_connection.py       # Boto3 S3 client (singleton)
│   ├── data_access/visa_data.py    # MongoDB collection → pandas DataFrame
│   ├── utils/main_utils.py         # YAML, dill, numpy helpers
│   ├── logger/__init__.py          # Timestamped file logger
│   └── exception/__init__.py       # Custom USvisaException with traceback
│
├── config/
│   ├── schema.yaml                 # Column definitions + encoder column lists
│   └── model.yaml                  # Model classes + hyperparameter search grids
│
├── templates/visa.html             # Jinja2 prediction form (Bootstrap 5)
├── static/css/style.css
├── artifact/                       # Auto-created; one folder per pipeline run
├── logs/                           # Timestamped .log file per run
├── notebooks/                      # EDA + feature engineering Jupyter notebooks
│   ├── 1_EDA_visa.ipynb
│   ├── 2_Feature_Engineering_and_Model_Training.ipynb
│   └── data_drift_detection_evidently.ipynb
│
├── app.py                          # FastAPI entry point
├── demo.py                         # Run training pipeline directly
├── Dockerfile
├── requirements.txt
├── setup.py                        # Installs visa package in editable mode
└── .github/workflows/cicd.yaml     # GitHub Actions CI/CD workflow
```

---

## 6. Data & Features

### Dataset

**Source:** [EasyVisa Dataset on Kaggle](https://www.kaggle.com/datasets/moro23/easyvisa-dataset)
**Size:** 25,480 rows × 12 columns

### Feature Dictionary

| Feature | Type | Values | Business Meaning |
|---------|------|--------|----------------|
| `continent` | Categorical | Asia, Africa, Europe, N.America, S.America, Oceania | Employee's home continent |
| `education_of_employee` | Ordinal | High School → Bachelor's → Master's → Doctorate | Highest qualification |
| `has_job_experience` | Binary | Y / N | Prior relevant work experience |
| `requires_job_training` | Binary | Y / N | Will employer provide job training |
| `no_of_employees` | Numeric | ~14,500 – 40,000+ | Employer headcount (company size proxy) |
| `yr_of_estab` | Numeric | 1897 – 2016 | Year employer was founded → derived to `company_age` |
| `region_of_employment` | Categorical | West, Northeast, South, Midwest, Island | US region where job is located |
| `prevailing_wage` | Numeric | ~600 – 320,000 | Government-mandated minimum wage for the role |
| `unit_of_wage` | Categorical | Hour, Week, Month, Year | Pay frequency |
| `full_time_position` | Binary | Y / N | Full-time vs part-time |
| **`case_status`** | **Binary (TARGET)** | **Certified / Denied** | **Outcome to predict** |

### Feature Engineering

```python
# Derived at transformation time (not in raw data):
company_age = CURRENT_YEAR - yr_of_estab

# Then drop the source columns:
drop_columns: [case_id, yr_of_estab]
```

> **Why?** `company_age` is more interpretable than a year. Older, more established companies tend to have stronger compliance records and higher approval rates.

---

## 7. ML Pipeline — Step by Step

### Architecture: Config → Component → Artifact

Every pipeline stage follows the same pattern:

```
Config Object  ──►  Component  ──►  Artifact Object
                     (does work)     (files + metadata passed to next stage)
```

---

### Step 1 — Data Ingestion

**Component:** `DataIngestion` | **Output:** `DataIngestionArtifact`

1. Connects to MongoDB Atlas using `MONGODB_URL` environment variable
2. Queries `visa_data` collection; exports all 25,480 rows as a DataFrame
3. Saves full dataset as `visa.csv` in `artifact/<timestamp>/data_ingestion/feature_store/`
4. Splits data 80/20 into `train.csv` and `test.csv`

---

### Step 2 — Data Validation

**Component:** `DataValidation` | **Output:** `DataValidationArtifact`

1. Checks column count matches `schema.yaml`
2. Verifies all required numerical and categorical columns exist in both sets
3. Runs Evidently's `DataDriftProfileSection` to detect statistical drift
4. Saves drift report as YAML. If drift is detected → pipeline stops

---

### Step 3 — Data Transformation

**Component:** `DataTransformation` | **Output:** `DataTransformationArtifact`

1. Engineers `company_age = CURRENT_YEAR - yr_of_estab`
2. Drops `case_id` and `yr_of_estab`
3. Builds a `ColumnTransformer` preprocessor:

   | Encoder | Applied To |
   |---------|-----------|
   | `OneHotEncoder` | `continent`, `unit_of_wage`, `region_of_employment` |
   | `OrdinalEncoder` | `has_job_experience`, `requires_job_training`, `full_time_position`, `education_of_employee` |
   | `PowerTransformer` (Yeo-Johnson) | `no_of_employees`, `company_age` |
   | `StandardScaler` | `no_of_employees`, `prevailing_wage`, `company_age` |

4. Maps target: `Certified → 0`, `Denied → 1`
5. Applies **SMOTEENN** on both train and test arrays to address class imbalance
6. Saves `preprocessing.pkl` (dill) and transformed arrays as `.npy` files

---

### Step 4 — Model Training

**Component:** `ModelTrainer` | **Output:** `ModelTrainerArtifact`

1. Loads transformed `train.npy` and `test.npy` arrays
2. Uses `neuro_mf` ModelFactory with `config/model.yaml` to initialise models:

   ```yaml
   # config/model.yaml (excerpt)
   model_selection:
     module_0:
       class: KNeighborsClassifier
       search_param_grid:
         n_neighbors: [3, 5, 9]
         weights: [uniform, distance]
     module_1:
       class: RandomForestClassifier
       search_param_grid:
         max_depth: [10, 15, 20]
         n_estimators: [3, 5, 9]
   ```

3. Runs `GridSearchCV` (cv=3) over all models and hyperparameter grids
4. Selects winner (highest CV score above `expected_accuracy` threshold of 0.6)
5. Wraps winner in `VisaModel(preprocessing_object, trained_model_object)`
6. Saves as `model.pkl` using dill; records F1, Precision, Recall

> **Result from training logs:** KNN (k=3, weighted distance) won with CV Score **95.37%**, F1 **0.822**, Precision **0.839**, Recall **0.805**

---

### Step 5 — Model Evaluation

**Component:** `ModelEvaluation` | **Output:** `ModelEvaluationArtifact`

1. Computes F1 of the newly trained model on the held-out test set
2. Downloads current production model from S3 (if one exists)
3. Accepts new model only if `F1_new > F1_production + 0.02`
4. If no production model exists yet → always accepted

---

### Step 6 — Model Pusher

**Component:** `ModelPusher` | **Output:** `ModelPusherArtifact`

1. If model was accepted in Step 5, uploads `model.pkl` to S3 bucket `visa2026`
2. FastAPI always loads from this S3 path → pushing automatically updates production

---

## 8. Model Performance

| Model | CV Score | F1 | Precision | Recall |
|-------|----------|-----|-----------|--------|
| **KNN (k=3, distance weights) ✅ Winner** | **95.37%** | **0.822** | **0.839** | **0.805** |
| RandomForestClassifier | < KNN | — | — | — |

### What the metrics mean

| Metric | Plain English |
|--------|--------------|
| **F1 Score** | Best single metric when classes are imbalanced — balances false positives and negatives |
| **Precision** | Of applications predicted as Denied, what % were actually Denied |
| **Recall** | Of all applications actually Denied, what % did the model catch |
| **CV Score** | Cross-validation score across 3 folds — guards against overfitting |

> **Note:** Adding XGBoost or CatBoost (see [Section 13](#13-how-to-improve-this-project)) can push ROC-AUC above 0.92 on this dataset.

---

## 9. How to Replicate — Full Setup Guide

### Prerequisites

- Python 3.8+
- Git
- Conda or `venv`
- AWS Account (free tier sufficient for S3)
- MongoDB Atlas Account (free M0 cluster sufficient)
- Docker Desktop (optional for local container testing)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sahatanmoyofficial/Global-Mobility-Application-Analyzer.git
cd Global-Mobility-Application-Analyzer
```

---

### Step 2 — Set Up Python Environment

```bash
# Option A: Conda (recommended)
conda create -n visa-app python=3.8 -y
conda activate visa-app

# Option B: venv
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
# This also installs the visa package in editable mode (-e .)
# Verify with: python -c "import visa; print('Package OK')"
```

---

### Step 4 — Set Up MongoDB Atlas

1. Create a free cluster at [cloud.mongodb.com](https://cloud.mongodb.com)
2. Create database: `VISA_APPLICATION_DATA`
3. Create collection: `visa_data`
4. Import the dataset:

```bash
# Download Visadataset.csv from Kaggle, then:
mongoimport \
  --uri "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/VISA_APPLICATION_DATA" \
  --collection visa_data \
  --type csv \
  --headerline \
  --file notebooks/Visadataset.csv

# OR use the mongoDB_test.ipynb notebook to upload via Python
```

5. Whitelist your IP address in **Atlas → Network Access**
6. Copy your connection string from **Atlas → Connect → Drivers**

---

### Step 5 — Set Up AWS

1. Create an IAM user with **AmazonS3FullAccess** (for model storage)
2. Create an S3 bucket named `visa2026` in region `eu-north-1`
3. Save the Access Key ID and Secret Access Key

> For production deployment, also create:
> - An EC2 instance (Ubuntu 22.04, `t2.medium` or larger)
> - An ECR repository named `visa2026`
> - Attach `AmazonEC2ContainerRegistryFullAccess` + `AmazonEC2FullAccess` to your IAM user

---

### Step 6 — Configure Environment Variables

```bash
# Linux / Mac — add to ~/.bashrc or ~/.zshrc then run: source ~/.bashrc
export MONGODB_URL="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=Cluster0"
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="eu-north-1"

# Windows PowerShell
$env:MONGODB_URL="mongodb+srv://..."
$env:AWS_ACCESS_KEY_ID="AKIA..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_DEFAULT_REGION="eu-north-1"
```

---

### Step 7 — (Optional) Extend the Model Config

You can add more models to `config/model.yaml` without touching any Python code:

```yaml
# Add to model_selection in config/model.yaml:
module_2:
  class: XGBClassifier
  module: xgboost
  params:
    n_estimators: 100
    max_depth: 6
  search_param_grid:
    n_estimators: [100, 200]
    max_depth: [4, 6, 8]
    learning_rate: [0.01, 0.1]
```

---

### Step 8 — Run the Training Pipeline

```bash
# Option A: via demo.py
python demo.py

# Option B: via the web app's /train endpoint (after starting app)
curl http://localhost:8080/train

# Watch logs in real time
tail -f logs/$(ls -t logs/ | head -1)
```

Artifacts are saved to `artifact/<MM_DD_YYYY_HH_MM_SS>/` — one timestamped folder per run.

---

## 10. Running the Application

### Local (no Docker)

```bash
# Ensure env vars are set, then:
python app.py

# App starts at http://localhost:8080
# API docs at http://localhost:8080/docs
```

### Local (with Docker)

```bash
# Build the image
docker build -t visa-app:latest .

# Run the container
docker run -d -p 8080:8080 \
  -e MONGODB_URL="$MONGODB_URL" \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  -e AWS_DEFAULT_REGION="eu-north-1" \
  visa-app:latest

# Open http://localhost:8080
```

### API Endpoints

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `GET` | `/` | Renders the prediction HTML form | HTML page |
| `POST` | `/` | Submits form data; returns prediction | HTML with `Visa-approved` or `Visa Not-Approved` |
| `GET` | `/train` | Triggers the full training pipeline | `Training successful !!` |
| `GET` | `/docs` | Auto-generated OpenAPI documentation | Swagger UI |

### Example Prediction (cURL)

```bash
curl -X POST http://localhost:8080/ \
  -F continent=Asia \
  -F "education_of_employee=Master's" \
  -F has_job_experience=Y \
  -F requires_job_training=N \
  -F no_of_employees=5000 \
  -F company_age=20 \
  -F region_of_employment=Northeast \
  -F prevailing_wage=85000 \
  -F unit_of_wage=Year \
  -F full_time_position=Y
```

---

## 11. CI/CD & Cloud Deployment

Every push to `main` triggers `.github/workflows/cicd.yaml`:

```
Developer ──► git push origin main
                     │
              GitHub Actions triggered
                     │
        ┌────────────┴─────────────┐
        │   Job 1: CI              │   Job 2: CD
        │   (ubuntu-latest)        │   (self-hosted EC2)
        │                          │
        │  1. Checkout             │  1. Checkout
        │  2. Configure AWS creds  │  2. Login to ECR
        │  3. Login to ECR         │  3. docker run -d -p 8080:8080
        │  4. docker build         │     (injects secrets as env vars)
        │  5. docker push → ECR    │  4. App live at http://<EC2-IP>:8080
        └──────────────────────────┘
```

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret key |
| `AWS_DEFAULT_REGION` | e.g. `eu-north-1` |
| `ECR_REPO` | ECR repository name, e.g. `visa2026` |
| `MONGODB_URL` | Full MongoDB Atlas connection string |

### EC2 Self-Hosted Runner Setup

```bash
# On your EC2 instance:
# 1. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu
newgrp docker

# 2. Register self-hosted runner
# GitHub → Settings → Actions → Runners → New self-hosted runner → Linux
# Follow the provided commands on your EC2 instance
```

---

## 12. Business Applications & Other Industries

### Immediate Use Cases (Immigration)

| User | Value Delivered |
|------|----------------|
| **Immigration Consultants** | Triage case load; focus time on borderline cases |
| **HR / Talent Acquisition** | Pre-screen candidates before spending on sponsorship |
| **Visa Applicants** | Self-assessment: "what can I change to improve my chances?" |
| **Law Firms** | Automated first-pass review; flag high-risk applications |
| **Government Agencies** | Anomaly detection — flag applications matching fraud patterns |

### Adjacent Industries

The same ML pattern (predicting a multi-factor administrative decision) transfers directly to:

| Industry | Analogous Problem | Key Features That Map |
|----------|------------------|----------------------|
| **Financial Services** | Loan / credit approval | Income, employment tenure, loan amount, region |
| **Insurance** | Claim approval / fraud detection | Policy type, claim amount, history |
| **Healthcare** | Prior authorisation | Diagnosis code, insurer, procedure, physician |
| **Government / Public Sector** | Permit & licence approval | Business size, region, application type |
| **Higher Education** | Scholarship / admission | GPA, region, field of study |
| **Real Estate** | Rental application screening | Income ratio, employment status, credit |
| **Procurement / Legal** | Contract approval routing | Vendor tier, contract value, risk flags |

---

## 13. How to Improve This Project

### 🧠 Model & ML Improvements

| Area | Priority | Recommended Action |
|------|----------|-------------------|
| **Models** | 🔴 High | Add XGBoost, LightGBM, CatBoost to `model.yaml` — these typically outperform KNN on tabular data |
| **Probability Output** | 🔴 High | Wrap classifier in `CalibratedClassifierCV` to output reliable probabilities (e.g. 73% chance) not just binary labels |
| **Explainability** | 🔴 High | Add SHAP values — return top 3 features driving each individual decision |
| **Fairness Audit** | 🔴 High | Audit for disparate impact by continent and education; use [Fairlearn](https://fairlearn.org/) |
| **Feature Engineering** | 🟡 Medium | Normalise wages to annual equivalent; add state-level labour statistics |
| **Train/Test Split** | 🟡 Medium | Use temporal split (train on older apps, test on recent) to prevent data leakage |
| **Hyperparameter Tuning** | 🟡 Medium | Replace GridSearchCV with [Optuna](https://optuna.org/) for smarter, faster search |
| **Ensemble** | 🟢 Low | Stack KNN + RandomForest + XGBoost with a meta-learner |

### 🏗️ MLOps & Infrastructure Improvements

| Area | Recommended Action |
|------|-------------------|
| **Experiment Tracking** | Integrate [MLflow](https://mlflow.org/) — log every run's parameters, metrics, and artifacts |
| **Model Registry** | Use MLflow Model Registry instead of raw S3 — adds staging/production lifecycle |
| **Data Versioning** | Add [DVC](https://dvc.org/) to track dataset versions alongside code |
| **Production Monitoring** | Deploy Evidently or Grafana to monitor prediction drift in production |
| **Auto Retraining** | Trigger retraining via webhook or Airflow DAG when drift exceeds threshold |
| **Testing** | Add `pytest` unit tests for each component; target 80% coverage |
| **Secrets Management** | Replace env vars with AWS Secrets Manager or HashiCorp Vault |
| **Scalability** | Move from EC2 to ECS Fargate or EKS for auto-scaling |

### 📦 Product Improvements

- Add **batch upload** — accept a CSV and return predictions for all rows
- Build an **admin dashboard** (Streamlit or React) showing prediction history and model metrics
- Add **SHAP waterfall charts** in the UI for individual prediction explanations
- Add **user authentication** (OAuth2 / JWT) for the API
- Localise for non-US systems: **UK Skilled Worker**, **Canadian Express Entry**, **Australian Skilled Independent**

---

## 14. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `MONGODB_URL not set` | Export the env var: `export MONGODB_URL="..."` |
| `pymongo` connection timeout | Whitelist your IP in MongoDB Atlas → Network Access |
| AWS credentials error | Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are exported |
| S3 bucket not found | Confirm bucket name is `visa2026` and region is `eu-north-1` |
| "No best model found" | Lower `MODEL_TRAINER_EXPECTED_SCORE` in `constants/__init__.py` from `0.6` |
| Drift detected, pipeline stops | Retrain with fresh data or investigate the drift report YAML |
| Docker build fails | Python 3.8 required — check `FROM` line in `Dockerfile` |
| Port 8080 already in use | `lsof -ti:8080 \| xargs kill -9` |
| EC2 cannot pull from ECR | Attach `AmazonEC2ContainerRegistryFullAccess` policy to EC2 IAM role |
| GitHub Actions deploy fails | Check EC2 self-hosted runner is online: Settings → Actions → Runners |

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| **Artifact** | The output of a pipeline component (files + metadata) passed to the next component |
| **CatBoost** | Gradient boosting library optimised for categorical features — recommended addition |
| **CI/CD** | Continuous Integration / Continuous Deployment — automated build, test, and deploy |
| **ColumnTransformer** | Scikit-learn class that applies different preprocessing to different column subsets |
| **dill** | Enhanced pickle library that can serialise complex Python objects including lambdas |
| **ECR** | Amazon Elastic Container Registry — private Docker image repository on AWS |
| **Evidently** | Open-source ML observability library for data and model monitoring |
| **F1 Score** | `2 × (Precision × Recall) / (Precision + Recall)` — balances false positives and negatives |
| **FastAPI** | Modern Python web framework with automatic OpenAPI docs |
| **GridSearchCV** | Exhaustive hyperparameter search using cross-validation |
| **KNN** | K-Nearest Neighbours — predicts by majority vote of k closest training examples |
| **LCA** | Labour Condition Application — the form US employers file to sponsor foreign workers |
| **MLOps** | DevOps practices applied to ML: versioning, CI/CD, monitoring, and retraining |
| **MongoDB Atlas** | Cloud-hosted MongoDB service used here as the raw data store |
| **neuro_mf** | Config-driven multi-model training library used for model selection |
| **OrdinalEncoder** | Encodes ordered categories as integers (e.g. High School=0, Bachelor=1) |
| **PowerTransformer** | Applies Yeo-Johnson transformation to make skewed data more Gaussian |
| **SHAP** | SHapley Additive exPlanations — explains individual model predictions |
| **SMOTEENN** | Synthetic Minority Over-sampling Technique + Edited Nearest Neighbours |
| **Uvicorn** | Lightning-fast ASGI server that runs FastAPI in production |
| **VisaModel** | Project wrapper class combining preprocessing pipeline + trained classifier |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | [github.com/sahatanmoyofficial](https://github.com/sahatanmoyofficial)

---

<div align="center">
<sub>Built with ❤️ using Python, FastAPI, MongoDB Atlas, AWS, and GitHub Actions</sub>
</div>
