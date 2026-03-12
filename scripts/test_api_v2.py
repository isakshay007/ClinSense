#!/usr/bin/env python3
"""
End-to-end verification script for ClinSense.
Tests the Cloud Run API health and prediction endpoints.
"""

import os
import requests
import json
import argparse

# Default Cloud Run URL - change via CLI or env var
DEFAULT_URL = "https://clinsense-api-xhyjwqbnza-uc.a.run.app"

def test_api(base_url):
    print(f"=== Testing ClinSense API ===")
    print(f"URL: {base_url}\n")
    
    # 1. Health Check
    print("[1/2] Checking API health (waiting up to 60s for cold start)...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=60)
        resp.raise_for_status()
        print(f"✅ Health: {resp.json().get('status', 'unknown')}")
    except Exception as e:
        print(f"❌ Health check failed (timed out or unreachable): {e}")
        return

    # 2. Prediction
    print("\n[2/2] Testing prediction endpoint...")
    sample_texts = [
        "Patient presents with chest pain and shortness of breath.",  # Cardiovascular / Pulmonary
        "MRI of the brain shows multiple white matter lesions.",     # Radiology / Neurology
        "Patient has acute abdominal pain and nausea."                # Gastroenterology
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}: '{text[:50]}...'")
        try:
            payload = {"text": text}
            resp = requests.post(f"{base_url}/predict", json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            spec = data.get("specialty", "N/A")
            conf = data.get("confidence", 0.0)
            print(f"✅ Result: {spec} (Confidence: {conf*100:.1f}%)")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ClinSense Cloud Run API")
    parser.add_argument("--url", default=os.getenv("CLINSENSE_API_URL", DEFAULT_URL), help="API base URL")
    args = parser.parse_args()
    
    test_api(args.url.rstrip("/"))
