<p align="center">
  <img src="https://img.shields.io/badge/ClinSense-Clinical%20Text%20Intelligence-38bdf8?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRoIGQ9Ik0zIDloMThNMzAxNWgxOE0xMiAzdjE4Ii8+PC9zdmc+" />
</p>

<h1 align="center">🏥 ClinSense</h1>
<p align="center">
  <strong>Clinical Text Intelligence & Entity Recognition</strong>
</p>
<p align="center">
  Classify medical notes by specialty • Extract drugs, diseases & chemicals • Production-ready API
</p>

<p align="center">
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-model-performance">Performance</a> •
  <a href="#-experiment-tracking--monitoring">Tracking</a> •
  <a href="#-deployment">Deployment</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Transformers-4.35+-FFD700?style=flat-square" />
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker" />
</p>

---

## Live Demo

**[→ Try ClinSense Live Demo](https://clinsense-demo.streamlit.app/)**

The demo lets you:
- **Classify** clinical notes into 8 medical specialties (BERT or TF-IDF+LR)
- **Extract** drugs, diseases & chemicals with scispaCy NER
- **Compare** model performance (Micro/Macro F1, per-class heatmaps)
- **Track** live metrics: predictions, latency, specialty distribution

### Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo, branch, and **Main file path**: `app/streamlit_app.py`
4. Click **Deploy** (Streamlit Cloud will install from `requirements.txt`)
5. After deployment, update the demo link above with your app URL (e.g. `https://your-repo-name.streamlit.app`)

> **Note:** Models must be in `models/` (BERT in `models/bert_finetuned/`, TF-IDF in `models/tfidf_lr.joblib`). For a fresh deploy, add the model files to your repo or use a different storage method.

---

##  Architecture

```
          ┌─────────────────────────────────────────────────────────────────────────────────┐
          │                         CLINSENSE — End-to-End Pipeline                         │
          └─────────────────────────────────────────────────────────────────────────────────┘

                                        ┌─────────────────────┐
                                        │   Clinical Note     │
                                        │   (Raw Text)        │
                                        └──────────┬──────────┘
                                                  │
                              ┌────────────────────┼────────────────────┐
                              │                    │                    │
                              ▼                    ▼                    ▼
                  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
                  │  BERT (LoRA)     │  │  TF-IDF + LR/SVM │  │  SciSpacy NER       │
                  │  Fine-tuned      │  │  Baseline        │  │  en_ner_bc5cdr_md   │
                  │  bert-base       │  │  sklearn         │  │  Chemicals/Diseases │
                  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────────┘
                          │                     │                     │
                          ▼                     ▼                     ▼
                  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
                  │  Specialty       │  │  Specialty       │  │  Entity Tags     │
                  │  + Confidence %  │  │  + Confidence %  │  │  Drugs • Diseases│
                  └──────────────────┘  └──────────────────┘  └──────────────────┘
                                                  │
                                                  ▼
                                        ┌─────────────────────┐
                                        │  Streamlit / API    │
                                        │  FastAPI • Docker   │
                                        └─────────────────────┘
```

### Data Flow

| Stage | Component | Output |
|-------|-----------|--------|
| **Input** | MTSamples (Kaggle) | ~5K transcriptions, 40+ specialties |
| **Filter** | `optimal_filter_and_train.py` | 8 classes, 1,911 samples |
| **Train** | TF-IDF + LR/SVM, BERT (Colab) | `models/tfidf_*.joblib`, `models/bert_finetuned/` |
| **Inference** | `predictor.py`, `baselines.py` | Specialty + confidence |
| **NER** | scispaCy `en_ner_bc5cdr_md` | Drugs (CHEMICAL), Diseases |

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/clinsense.git
cd clinsense
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Streamlit Demo (Local)

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Open **http://localhost:8501**

### 3. Run the FastAPI Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API docs: **http://localhost:8000/docs**

---

## Model Performance

| Model | Micro F1 | Macro F1 | Notes |
|-------|----------|----------|-------|
| **BERT (fine-tuned)** | **71.1%** | **70.1%** | Best performer |
| TF-IDF + Logistic Regression | 67.9% | 68.3% | Fast baseline |
| TF-IDF + SVM | 65.9% | 66.3% | Linear kernel |

**8 specialties:** Cardiovascular/Pulmonary, Gastroenterology, Neurology, OB/GYN, Orthopedic, Radiology, SOAP/Chart, Urology

**Dataset:** MTSamples (filtered to 1,911 samples)

---

##  Project Structure

```
ClinSense/
├── app/
│   ├── main.py              # FastAPI app (predict, health, drift)
│   └── streamlit_app.py     # Streamlit demo dashboard
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                 # MTSamples CSV (Kaggle)
│   └── processed/           # Filtered & preprocessed
├── models/
│   ├── bert_finetuned/      # BERT model (from Colab)
│   ├── tfidf_lr.joblib
│   └── tfidf_svm.joblib
├── notebooks/
│   └── biobert_lora_clinsense.ipynb   # Colab fine-tuning
├── scripts/
│   ├── train_baselines.py
│   ├── predict_bert.py
│   ├── optimal_filter_and_train.py
│   ├── extract_entities.py
│   ├── deploy_aws.sh
│   └── test_api.sh
├── src/
│   ├── services/predictor.py    # BERT prediction + proba
│   ├── models/baselines.py      # TF-IDF LR/SVM
│   ├── ner/scispacy_ner.py      # Entity extraction
│   ├── monitoring/drift.py      # Evidently AI
│   └── data/loader.py
├── aws/                    # ECS task definition, ECR
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

##  Usage

### Train TF-IDF Baselines

```bash
python scripts/train_baselines.py --model both --download --wandb --mlflow
```

### BERT Fine-Tuning (Google Colab)

1. Create filtered dataset: `python scripts/optimal_filter_and_train.py`
2. Upload `data/processed/mtsamples_classification_filtered.csv` to Colab
3. Open `notebooks/biobert_lora_clinsense.ipynb`, enable GPU
4. Download `clinsense_bert_final` → `models/bert_finetuned/`

### Predict (CLI)

```bash
python scripts/predict_bert.py "Patient presents with knee pain, history of osteoarthritis..."
python scripts/predict_bert.py --file path/to/note.txt
```

### Entity Extraction

```bash
pip install spacy scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
python scripts/extract_entities.py --download
```

---

## 📊 Experiment Tracking & Monitoring

ClinSense uses three tools for experiment tracking and production monitoring. Here’s what each one provides in this project:

| Tool | When used | What it gives |
|------|-----------|---------------|
| **W&B** | During training (`--wandb`) | Logs metrics (F1, precision, recall), hyperparameters, and run config to the W&B dashboard. Useful for comparing runs and sharing results. |
| **MLflow** | During training (`--mlflow`) | Stores runs in `mlruns/`, logs params and metrics, saves model artifacts, and registers the best model as `clinsense-best`. Run `mlflow ui` to browse runs. |
| **Evidently AI** | Production API (`POST /monitor/drift`) | Compares reference vs current text distributions to detect data drift. Returns `drift_detected`, `share_of_drifted_columns`, mean text lengths, and sample counts. |

**Example MLflow output:** `macro_f1: 0.66`, `micro_f1: 0.65`, model artifacts, registered model versions.

**Example Evidently drift output:** `{"drift_detected": false, "reference_mean_length": 2814, "current_mean_length": 1582, "reference_count": 100, "current_count": 100}`.

**W&B setup:** `wandb login` or set `WANDB_API_KEY` in `.env`. Use `WANDB_MODE=offline` to log locally without login.

---

## Deployment

### Streamlit Cloud (Demo) — No AWS required

For the interactive demo dashboard, deploy to Streamlit Cloud only:

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, set `app/streamlit_app.py` as main file
4. Deploy → get your live demo URL

Uses `runtime.txt` (Python 3.11) for compatibility.

### Docker (FastAPI locally)

```bash
docker build -t clinsense-api .
docker run -p 8000:8000 clinsense-api
```

### AWS ECS (FastAPI API only) — Optional

**When needed:** Only if you want to host the **FastAPI REST API** (`/predict`, `/health`, `/monitor/drift`) in production. The Streamlit demo does **not** use AWS.

**Prerequisites:**
- AWS CLI configured (`aws configure`)
- ECS cluster created
- `ecsTaskExecutionRole` IAM role (ECR pull + CloudWatch logs)
- `models/bert_finetuned/` present before `docker build`

```bash
# Deploy (builds image, pushes to ECR, registers task def, updates service)
./scripts/deploy_aws.sh

# Optional env vars
AWS_REGION=us-east-1 ECS_CLUSTER=clinsense ECS_SERVICE=clinsense-api ./scripts/deploy_aws.sh
```

**First-time setup:** Create ECS cluster and service with a load balancer, then run the deploy script.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/predict` | `{"text": "..."}` → `{"specialty": "...", "confidence": 0.85}` |
| POST | `/monitor/drift` | Evidently drift: reference vs current texts |

---

##  License

MIT

---

<p align="center">
  <strong>ClinSense</strong> — Clinical Text Intelligence & Entity Recognition
</p>
