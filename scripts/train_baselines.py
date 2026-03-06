#!/usr/bin/env python3
"""
Train TF-IDF + Logistic Regression and TF-IDF + SVM baselines for ClinSense.

Tracks experiments in W&B and registers best model in MLflow.
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_data_paths
from src.data.loader import load_mtsamples, load_mtsamples_from_csv, prepare_classification_data
from src.evaluation import compute_classification_metrics, log_metrics_to_mlflow, log_metrics_to_wandb
from src.models.baselines import TfidfLogisticRegression, TfidfSVM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lr", "svm", "both"], default="both")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--mlflow", action="store_true", help="Log to MLflow")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle")
    args = parser.parse_args()

    config = load_config()
    raw_dir, processed_dir = get_data_paths(config)
    data_cfg = config["data"]

    # Prefer optimal-filtered dataset if available
    filtered_path = processed_dir / data_cfg.get("filtered_file", "mtsamples_classification_filtered.csv")
    if data_cfg.get("use_filtered", False) and filtered_path.exists():
        df = load_mtsamples_from_csv(filtered_path)
    else:
        df = load_mtsamples(
            raw_dir=raw_dir,
            download_if_missing=args.download,
            dataset=config["data"]["kaggle_dataset"],
            min_samples_per_class=data_cfg.get("min_samples_per_class", 100),
            top_n_classes=data_cfg.get("top_n_classes", 10),
        )
    print(f"Classes: {df['medical_specialty'].nunique()}, Samples: {len(df)}")
    print("Train distribution:\n", df["medical_specialty"].value_counts())

    X_train, X_val, X_test, y_train, y_val, y_test, label_names = prepare_classification_data(
        df,
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_state=config["data"]["random_state"],
    )

    tfidf_cfg = config["tfidf"]
    lr_cfg = config["logistic_regression"]
    svm_cfg = config["svm"]

    if args.wandb:
        import wandb
        wandb.init(project=config["wandb"]["project"], config={
            "model": args.model,
            "tfidf_max_features": tfidf_cfg["max_features"],
            "test_size": config["data"]["test_size"],
        })

    if args.mlflow:
        import mlflow
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

    best_metric = -1.0
    best_run_id = None

    def run_model(name: str, model_class, model_kwargs: dict):
        nonlocal best_metric, best_run_id
        ctx = mlflow.start_run(run_name=name) if args.mlflow else nullcontext()
        with ctx:
            m = model_class(**model_kwargs)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            metrics = compute_classification_metrics(y_test, y_pred)
            print(f"\n--- {name} ---")
            if name == "tfidf_svm":
                from collections import Counter
                print(f"  Prediction distribution: {dict(Counter(y_pred.tolist()))}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            if args.wandb:
                log_metrics_to_wandb(metrics, prefix=f"{name}/")
            model_path = Path(__file__).parent.parent / "models" / f"{name}.joblib"
            m.save(model_path)
            if args.mlflow:
                mlflow.log_params(model_kwargs)
                log_metrics_to_mlflow(metrics)
                mlflow.sklearn.log_model(m.pipeline, "model")
                mlflow.log_artifact(str(model_path))

            if metrics["macro_f1"] > best_metric:
                best_metric = metrics["macro_f1"]
                if args.mlflow:
                    best_run_id = mlflow.active_run().info.run_id
        return m

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    lr_kwargs = {
        "max_features": tfidf_cfg["max_features"],
        "ngram_range": tuple(tfidf_cfg["ngram_range"]),
        "max_df": tfidf_cfg["max_df"],
        "min_df": tfidf_cfg["min_df"],
        "sublinear_tf": tfidf_cfg["sublinear_tf"],
        "C": lr_cfg["C"],
        "max_iter": lr_cfg["max_iter"],
        "class_weight": lr_cfg["class_weight"],
    }
    svm_kwargs = {
        "max_features": tfidf_cfg["max_features"],
        "ngram_range": tuple(tfidf_cfg["ngram_range"]),
        "max_df": tfidf_cfg["max_df"],
        "min_df": tfidf_cfg["min_df"],
        "sublinear_tf": tfidf_cfg["sublinear_tf"],
        "C": svm_cfg["C"],
        "max_iter": svm_cfg.get("max_iter", 5000),
        "class_weight": svm_cfg["class_weight"],
    }

    if args.model in ("lr", "both"):
        run_model("tfidf_lr", TfidfLogisticRegression, lr_kwargs)
    if args.model in ("svm", "both"):
        run_model("tfidf_svm", TfidfSVM, svm_kwargs)

    if args.mlflow and best_run_id:
        import mlflow
        mlflow.register_model(f"runs:/{best_run_id}/model", "clinsense-best")


if __name__ == "__main__":
    main()
