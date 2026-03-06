#!/bin/bash
# Test ClinSense API (run after: uvicorn app.main:app --port 8000)
BASE=${1:-http://localhost:8000}

echo "=== Testing ClinSense API ==="
echo ""

echo "1. Root"
curl -s "$BASE/" | python3 -m json.tool
echo ""

echo "2. Health"
curl -s "$BASE/health" | python3 -m json.tool
echo ""

echo "3. Predict"
curl -s -X POST "$BASE/predict" -H "Content-Type: application/json" \
  -d '{"text":"Patient presents with acute chest pain. ECG shows ST elevation. Troponin elevated."}' | python3 -m json.tool
echo ""

echo "4. Monitor drift"
curl -s -X POST "$BASE/monitor/drift" -H "Content-Type: application/json" \
  -d '{"reference_texts":["Short note","Another short"],"current_texts":["Short note","Very long note that might indicate drift"]}' | python3 -m json.tool
echo ""

echo "=== All tests done ==="
