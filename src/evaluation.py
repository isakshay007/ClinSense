"""Evaluation metrics for ClinSense: Micro F1, Macro F1, Precision, Recall, Hamming Loss."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    average: str | None = "both",
    zero_division: str | int = 0,
) -> dict[str, float]:
    """
    Compute Micro F1, Macro F1, Precision, Recall, and Hamming Loss.
    
    For multi-class, Hamming Loss is 1 - accuracy (single label).
    For multi-label, Hamming Loss would be per-label loss.
    Here we use multi-class single-label.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {}
    
    # Micro-averaged (global)
    metrics["micro_f1"] = float(
        f1_score(y_true, y_pred, average="micro", zero_division=zero_division)
    )
    metrics["micro_precision"] = float(
        precision_score(y_true, y_pred, average="micro", zero_division=zero_division)
    )
    metrics["micro_recall"] = float(
        recall_score(y_true, y_pred, average="micro", zero_division=zero_division)
    )
    
    # Macro-averaged (per-class, then average)
    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=zero_division)
    )
    metrics["macro_precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=zero_division)
    )
    metrics["macro_recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=zero_division)
    )
    
    # Overall precision & recall (same as micro for single-label)
    metrics["precision"] = metrics["micro_precision"]
    metrics["recall"] = metrics["micro_recall"]
    
    # Hamming Loss: for single-label multi-class, 1 - accuracy
    # sklearn's hamming_loss expects binary/multi-label; for multi-class we use 1 - accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["hamming_loss"] = 1.0 - metrics["accuracy"]
    
    return metrics


def log_metrics_to_wandb(metrics: dict[str, float], prefix: str = "") -> None:
    """Log metrics dict to Weights & Biases."""
    try:
        import wandb
        logged = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        wandb.log(logged)
    except ImportError:
        pass


def log_metrics_to_mlflow(metrics: dict[str, float]) -> None:
    """Log metrics dict to MLflow."""
    try:
        import mlflow
        mlflow.log_metrics(metrics)
    except ImportError:
        pass
