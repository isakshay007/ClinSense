# ClinSense: Enterprise Clinical Text Intelligence

ClinSense is a production-grade clinical intelligence system designed for automated medical specialty classification and entity extraction. The platform implements a high-performance hybrid-cloud architecture to bridge large-scale data processing with real-time inference.

## Primary System Features

- Synchronized Production Data Flow: Complete parity between Databricks PySpark analytical pipelines and local inference modules.
- Advanced Language Modeling: Fine-tuned BERT-base-uncased architecture localized for medical transcription analysis.
- Entity Recognition Engine: Integration of scispaCy for high-fidelity extraction of drugs, diseases, and chemical compounds.
- Enterprise API Layer: Serverless FastAPI implementation hosted on GCP Cloud Run with integrated health monitoring.
- Strategic Visualization: High-performance Streamlit dashboard featuring lazy-loading model initialization and real-time latency tracking.

## Technical Architecture

The ClinSense ecosystem is structured into three specialized architectural layers:

### 1. Analytical Layer (Databricks)
The system utilizes Databricks for large-scale ingestion and preprocessing of the MTSamples dataset. The PySpark pipeline enforces rigorous data governance, including minimum sample thresholds and word density filters, outputting standardized Parquet files for training and validation.

### 2. Inference and Serving Layer (GCP Cloud Run)
Real-time intelligence is delivered via a containerized FastAPI application. Hosted on Google Cloud Run, this layer provides a scalable REST API for specialty classification. It includes automated data drift detection and system health endpoints.

### 3. Presentation and User Experience Layer (Streamlit)
A professional medical dashboard facilitates human-in-the-loop interaction. Features include real-time transcription analysis, interactive entity visualization, and comparative model performance metrics.

## Data Governance and Synchronization

System reliability is maintained through strict logic synchronization between all environments. The following governance standards are enforced globally via centralized configuration:

- Minimum Samples per Specialty Class: 150
- Minimum Average Word Count per Transcription: 100
- Maximum Input Token Length: 512
- Support for 8 Pre-synchronized Clinical Specialties

Synchronization is verified using the audit utility located at scripts/diagnose_data.py.

## Model Performance Metrics

Classification accuracy is benchmarked against established baselines to ensure optimal production performance.

| Model Architecture | Micro F1 Score | Macro F1 Score | Status |
| :--- | :--- | :--- | :--- |
| Fine-tuned BERT | 71.1% | 70.1% | Active Production |
| TF-IDF + Logistic Regression | 67.9% | 68.3% | Analytical Baseline |
| TF-IDF + SVM | 65.9% | 66.3% | Analytical Baseline |

## Implementation and Deployment

### Environment Initialization
1. Initialize the Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Data Parity Audit
Execute the diagnostic script to verify local environment alignment with production governance standards:
```bash
python scripts/diagnose_data.py
```

### Strategic Deployment
The system is optimized for containerized serverless deployment. Infrastructure requirements are defined in gcp/cloud-run.yaml.

Deploy the production stack:
```bash
./scripts/deploy_gcp.sh
```

### End-to-End Verification
Validate the live production API and inference accuracy:
```bash
python scripts/test_api_v2.py --url https://clinsense-api-xhyjwqbnza-uc.a.run.app
```

## System Structure

```text
ClinSense/
+-- databricks/
|   +-- preprocess_pipeline.py    # Production PySpark data pipeline
+-- app/
|   +-- streamlit_app.py         # Advanced clinical dashboard
|   +-- main.py                  # Production FastAPI implementation
+-- src/
|   +-- services/predictor.py     # BERT inference orchestration
|   +-- data/loader.py            # Synchronized data ingestion logic
|   +-- ner/scispacy_ner.py       # Clinical entity extraction logic
+-- scripts/
|   +-- deploy_gcp.sh            # Automated GCP deployment infrastructure
|   +-- diagnose_data.py         # Enterprise data parity audit tool
|   +-- predict_bert.py          # Command-line inference utility
+-- config/
|   +-- config.yaml              # Global system configuration
```

## License
Licensed under the MIT License.
