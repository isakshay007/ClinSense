#!/bin/bash
# Deploy ClinSense API to GCP Artifact Registry + Cloud Run
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Docker installed and running
#   - GCP project with Cloud Run API enabled:
#       gcloud services enable run.googleapis.com artifactregistry.googleapis.com
#   - models/bert_finetuned/ must exist locally before docker build
#   - (Optional) Set --allow-unauthenticated below if you want a public endpoint
#
# Usage:
#   ./scripts/deploy_gcp.sh
#   GCP_PROJECT=my-proj GCP_REGION=us-east1 ./scripts/deploy_gcp.sh

set -euo pipefail

# ── Config (override via env vars) ────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source .env if it exists
if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

GCP_PROJECT="${GCP_PROJECT:-YOUR_GCP_PROJECT_ID}"
GCP_REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-clinsense-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"


IMAGE_NAME="gcr.io/${GCP_PROJECT}/${SERVICE_NAME}:${IMAGE_TAG}"

echo "=== ClinSense GCP Cloud Run Deploy ==="
echo "Project:  ${GCP_PROJECT}"
echo "Region:   ${GCP_REGION}"
echo "Service:  ${SERVICE_NAME}"
echo "Image:    ${IMAGE_NAME}"
echo ""

# ── Guard: models must exist before baking into image ─────────────────────────
if [ ! -d "${PROJECT_ROOT}/models/bert_finetuned" ]; then
  echo "ERROR: models/bert_finetuned/ not found."
  echo "  Download the fine-tuned model from Colab and place it there first."
  exit 1
fi

# ── 1. Configure Docker to authenticate with GCR ──────────────────────────────
echo "[1/5] Configuring Docker for GCR..."
gcloud auth configure-docker --quiet

# ── 2. Build Docker image ──────────────────────────────────────────────────────
echo "[2/5] Building Docker image for linux/amd64..."
cd "${PROJECT_ROOT}"
docker build --platform linux/amd64 -t "${IMAGE_NAME}" .

# ── 3. Push to Google Container Registry ──────────────────────────────────────
echo "[3/5] Pushing image to GCR..."
docker push "${IMAGE_NAME}"

# ── 4. Deploy to Cloud Run ─────────────────────────────────────────────────────
echo "[4/5] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_NAME}" \
  --region "${GCP_REGION}" \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 5 \
  --port 8000 \
  --concurrency 1 \
  --timeout 300 \
  --set-env-vars "CLINSENSE_PRELOAD=true,CLINSENSE_MODEL_PATH=/app/models/bert_finetuned" \
  --no-allow-unauthenticated \
  --project "${GCP_PROJECT}"

# ── 5. Print service URL ───────────────────────────────────────────────────────
echo "[5/5] Fetching service URL..."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region "${GCP_REGION}" \
  --project "${GCP_PROJECT}" \
  --format "value(status.url)")

echo ""
echo "=== Deploy complete! ==="
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test it:"
echo "  # Health check (requires auth token — see note below)"
echo "  TOKEN=\$(gcloud auth print-identity-token)"
echo "  curl -H \"Authorization: Bearer \$TOKEN\" ${SERVICE_URL}/health"
echo ""
echo "  # Predict"
echo "  curl -H \"Authorization: Bearer \$TOKEN\" \\"
echo "       -H \"Content-Type: application/json\" \\"
echo "       -X POST ${SERVICE_URL}/predict \\"
echo "       -d '{\"text\": \"Patient presents with chest pain and ST elevation.\"}'"
echo ""
echo "  # Or use test_api.sh (set BASE to: '${SERVICE_URL}' + add auth header)"
echo "  BASE=${SERVICE_URL} ./scripts/test_api.sh"
echo ""
echo "NOTE: The service is deployed with --no-allow-unauthenticated."
echo "  To make it public, rerun with --allow-unauthenticated instead."
echo "  To grant access to a service account:"
echo "    gcloud run services add-iam-policy-binding ${SERVICE_NAME} \\"
echo "      --region=${GCP_REGION} --member=serviceAccount:SA@${GCP_PROJECT}.iam.gserviceaccount.com \\"
echo "      --role=roles/run.invoker"
