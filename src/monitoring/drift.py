"""Evidently AI data drift monitoring."""

import pandas as pd


def compute_drift_report(
    reference_texts: list[str],
    current_texts: list[str],
    column_name: str = "text",
) -> dict:
    """
    Compute data drift between reference and current text data.
    Uses text length as numeric feature. Falls back to basic stats if Evidently unavailable.
    """
    ref_lens = [len(t) for t in reference_texts]
    cur_lens = [len(t) for t in current_texts]
    ref_mean = sum(ref_lens) / len(ref_lens) if ref_lens else 0
    cur_mean = sum(cur_lens) / len(cur_lens) if cur_lens else 0

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        ref_df = pd.DataFrame({"text_length": ref_lens})
        cur_df = pd.DataFrame({"text_length": cur_lens})
        report = Report([DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        result = report.as_dict() if hasattr(report, "as_dict") else {}
        metrics = result.get("metrics", [])
        drift_share = None
        for m in metrics:
            if "result" in m and "dataset_drift" in m.get("result", {}):
                drift_share = m["result"].get("share_of_drifted_columns", 0)
                break
        return {
            "drift_detected": drift_share is not None and drift_share > 0,
            "share_of_drifted_columns": drift_share,
            "reference_count": len(reference_texts),
            "current_count": len(current_texts),
            "reference_mean_length": ref_mean,
            "current_mean_length": cur_mean,
        }
    except ImportError:
        # Fallback: simple length-based drift
        drift = abs(cur_mean - ref_mean) / (ref_mean + 1) > 0.5
        return {
            "drift_detected": drift,
            "share_of_drifted_columns": 1.0 if drift else 0.0,
            "reference_count": len(reference_texts),
            "current_count": len(current_texts),
            "reference_mean_length": ref_mean,
            "current_mean_length": cur_mean,
            "note": "Evidently not installed; using length-based heuristic",
        }
