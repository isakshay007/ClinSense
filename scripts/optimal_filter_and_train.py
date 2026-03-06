#!/usr/bin/env python3
"""
Deep dataset analysis + optimal filtering + retrain.
Steps 1-5: Text quality, separability, filtering, verification, retrain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, get_data_paths
from src.data.loader import load_mtsamples


def word_count(text: str) -> int:
    return len(str(text).split())


def main():
    config = load_config()
    raw_dir, processed_dir = get_data_paths(config)

    # Load ALL 40 classes (no filtering)
    df = load_mtsamples(
        raw_dir=raw_dir,
        download_if_missing=False,
        min_samples_per_class=1,
        top_n_classes=None,
    )

    col = "medical_specialty"
    text_col = "transcription"
    df["word_count"] = df[text_col].apply(word_count)

    # =========================================================================
    # STEP 1: TEXT QUALITY PER SPECIALTY
    # =========================================================================
    print("=" * 80)
    print("STEP 1: TEXT QUALITY PER SPECIALTY")
    print("=" * 80)

    quality_rows = []
    for spec in sorted(df[col].unique()):
        sub = df[df[col] == spec]
        texts = sub[text_col]
        wc = sub["word_count"]
        avg_words = wc.mean()
        std_words = wc.std() if len(wc) > 1 else 0
        n_dup = texts.duplicated().sum()
        n_short = (wc < 50).sum()
        weak_signal = avg_words < 100
        quality_rows.append({
            "specialty": spec,
            "count": len(sub),
            "avg_words": avg_words,
            "std_words": std_words,
            "n_duplicates": n_dup,
            "n_short_50": n_short,
            "weak_signal": weak_signal,
        })

    quality_df = pd.DataFrame(quality_rows).sort_values("count", ascending=False)

    for _, r in quality_df.iterrows():
        flag = " [WEAK SIGNAL]" if r["weak_signal"] else ""
        print(f"{r['specialty'][:45]:45s} | n={r['count']:4d} | avg={r['avg_words']:6.1f} words | "
              f"std={r['std_words']:5.1f} | dup={r['n_duplicates']:3d} | short50={r['n_short_50']:3d}{flag}")

    weak_specs = set(quality_df[quality_df["weak_signal"]]["specialty"].tolist())
    print(f"\nSpecialties with avg < 100 words (weak signal): {weak_specs}")

    # =========================================================================
    # STEP 2: CLASS SEPARABILITY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: CLASS SEPARABILITY (TF-IDF 5k + LR, per-class F1)")
    print("=" * 80)

    X = df[text_col]
    y = df[col]
    label_names = sorted(y.unique())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), max_df=0.95, min_df=2, sublinear_tf=True)
    X_tr_vec = vec.fit_transform(X_tr)
    X_te_vec = vec.transform(X_te)

    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42)
    lr.fit(X_tr_vec, y_tr)
    y_pred = lr.predict(X_te_vec)

    per_class_f1 = f1_score(y_te, y_pred, labels=label_names, average=None, zero_division=0)
    f1_by_spec = dict(zip(label_names, per_class_f1))

    print("\nPer-class F1 (all 40 classes):")
    for spec in sorted(label_names, key=lambda s: -f1_by_spec.get(s, 0)):
        f1 = f1_by_spec.get(spec, 0)
        status = "SEPARABLE" if f1 > 0.40 else ("CONFUSED" if f1 < 0.20 else "moderate")
        print(f"  {spec[:50]:50s} F1={f1:.3f}  [{status}]")

    separable = [s for s in label_names if f1_by_spec.get(s, 0) > 0.40]
    confused = [s for s in label_names if f1_by_spec.get(s, 0) < 0.20]
    print(f"\nSeparable (F1>0.40): {len(separable)} classes")
    print(f"Confused (F1<0.20): {len(confused)} classes: {confused}")

    # Confusion pairs: which classes get confused most
    cm = confusion_matrix(y_te, y_pred, labels=label_names)
    n_classes = len(label_names)
    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                total_i = cm[i, :].sum()
                pct = 100 * cm[i, j] / total_i if total_i > 0 else 0
                pairs.append((label_names[i], label_names[j], int(cm[i, j]), pct))
    pairs.sort(key=lambda x: -x[3])

    print("\nTop confusion pairs (true -> predicted, % of true class):")
    for true_s, pred_s, cnt, pct in pairs[:20]:
        print(f"  {true_s[:30]:30s} -> {pred_s[:30]:30s}  {cnt:3d} ({pct:5.1f}%)")

    # Pairs that confuse > 30%
    high_confusion_pairs = [(a, b, p) for a, b, c, p in pairs if p > 30]
    print(f"\nPairs with >30% confusion: {high_confusion_pairs[:15]}")

    # =========================================================================
    # STEP 3: OPTIMAL FILTERING
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: OPTIMAL FILTERING")
    print("=" * 80)

    count_by_spec = df[col].value_counts()
    avg_words_by_spec = df.groupby(col)["word_count"].mean().to_dict()

    # Filters
    drop_reasons = {}
    keep = set(label_names)

    for spec in label_names:
        reasons = []
        if count_by_spec.get(spec, 0) < 150:
            reasons.append("< 150 samples")
        if f1_by_spec.get(spec, 0) < 0.25:
            reasons.append("F1 < 0.25")
        if avg_words_by_spec.get(spec, 0) < 100:
            reasons.append("avg words < 100")
        if reasons:
            drop_reasons[spec] = reasons
            keep.discard(spec)

    # Merge/drop for high-confusion pairs: drop smaller class
    for true_s, pred_s, pct in high_confusion_pairs:
        if pct > 30 and true_s in keep and pred_s in keep:
            c_true = count_by_spec.get(true_s, 0)
            c_pred = count_by_spec.get(pred_s, 0)
            drop = true_s if c_true < c_pred else pred_s
            keep.discard(drop)
            drop_reasons[drop] = drop_reasons.get(drop, []) + [f"confused with {true_s if drop == pred_s else pred_s} >30%"]

    keep = sorted(keep)
    print(f"Dropped: {[s for s in label_names if s not in keep]}")
    for s, reasons in drop_reasons.items():
        print(f"  - {s}: {reasons}")

    df_filtered = df[df[col].isin(keep)].copy().reset_index(drop=True)

    # =========================================================================
    # STEP 4: VERIFY FILTERED DATASET
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: VERIFY FILTERED DATASET")
    print("=" * 80)

    vc = df_filtered[col].value_counts()
    total_samples = len(df_filtered)
    total_classes = len(keep)
    min_per_class = vc.min()
    max_per_class = vc.max()
    avg_per_class = vc.mean()
    balance_ratio = max_per_class / min_per_class if min_per_class > 0 else float("inf")

    print(f"Total samples: {total_samples}")
    print(f"Total classes: {total_classes}")
    print(f"Min samples/class: {min_per_class}")
    print(f"Max samples/class: {max_per_class}")
    print(f"Avg samples/class: {avg_per_class:.1f}")
    print(f"Balance ratio (largest/smallest): {balance_ratio:.1f}x")

    print("\nSurviving specialties with count, avg words, per-class F1:")
    for spec in keep:
        cnt = vc[spec]
        avg_w = avg_words_by_spec.get(spec, 0)
        f1 = f1_by_spec.get(spec, 0)
        print(f"  {spec[:45]:45s} | n={cnt:4d} | avg_words={avg_w:6.1f} | F1={f1:.3f}")

    print("\n2 sample texts per class (first 200 chars):")
    for spec in keep:
        samples = df_filtered[df_filtered[col] == spec][text_col].head(2)
        for i, txt in enumerate(samples):
            preview = str(txt)[:200].replace("\n", " ")
            print(f"\n  [{spec}] sample {i+1}: {preview}...")

    # =========================================================================
    # STEP 5: RETRAIN WITH 70/15/15 SPLIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: RETRAIN (70/15/15 split)")
    print("=" * 80)

    X = df_filtered[text_col]
    y = df_filtered[col]

    # 70 train, 30 rest -> split 30 into 15 val, 15 test
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # TF-IDF + LR
    vec_lr = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), max_df=0.95, min_df=2, sublinear_tf=True)
    X_tr_vec = vec_lr.fit_transform(X_train)
    X_te_vec = vec_lr.transform(X_test)

    lr_model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42)
    lr_model.fit(X_tr_vec, y_train)
    y_pred_lr = lr_model.predict(X_te_vec)

    micro_f1_lr = f1_score(y_test, y_pred_lr, average="micro", zero_division=0)
    macro_f1_lr = f1_score(y_test, y_pred_lr, average="macro", zero_division=0)
    weighted_f1_lr = f1_score(y_test, y_pred_lr, average="weighted", zero_division=0)

    print("\n--- TF-IDF + LR ---")
    print(f"  Micro F1:   {micro_f1_lr:.4f}")
    print(f"  Macro F1:   {macro_f1_lr:.4f}")
    print(f"  Weighted F1: {weighted_f1_lr:.4f}")
    print("\nClassification Report (LR):")
    print(classification_report(y_test, y_pred_lr, zero_division=0))

    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=keep)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_lr, cmap="Blues")
    ax.set_xticks(range(len(keep)))
    ax.set_yticks(range(len(keep)))
    ax.set_xticklabels(keep, rotation=45, ha="right")
    ax.set_yticklabels(keep)
    for i in range(len(keep)):
        for j in range(len(keep)):
            ax.text(j, i, str(cm_lr[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Confusion Matrix: TF-IDF + Logistic Regression")
    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "outputs" / "confusion_matrix_lr.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to {out_path}")

    # TF-IDF + SVM
    svm_model = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, dual="auto", random_state=42)
    svm_model.fit(X_tr_vec, y_train)
    y_pred_svm = svm_model.predict(X_te_vec)

    micro_f1_svm = f1_score(y_test, y_pred_svm, average="micro", zero_division=0)
    macro_f1_svm = f1_score(y_test, y_pred_svm, average="macro", zero_division=0)
    weighted_f1_svm = f1_score(y_test, y_pred_svm, average="weighted", zero_division=0)

    print("\n--- TF-IDF + SVM ---")
    print(f"  Micro F1:   {micro_f1_svm:.4f}")
    print(f"  Macro F1:   {macro_f1_svm:.4f}")
    print(f"  Weighted F1: {weighted_f1_svm:.4f}")
    print("\nClassification Report (SVM):")
    print(classification_report(y_test, y_pred_svm, zero_division=0))

    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=keep)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_svm, cmap="Greens")
    ax.set_xticks(range(len(keep)))
    ax.set_yticks(range(len(keep)))
    ax.set_xticklabels(keep, rotation=45, ha="right")
    ax.set_yticklabels(keep)
    for i in range(len(keep)):
        for j in range(len(keep)):
            ax.text(j, i, str(cm_svm[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Confusion Matrix: TF-IDF + SVM")
    plt.tight_layout()
    out_path_svm = Path(__file__).parent.parent / "outputs" / "confusion_matrix_svm.png"
    plt.savefig(out_path_svm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to {out_path_svm}")

    # Final summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print("| Model       | Classes | Train Samples | Micro F1 | Macro F1 |")
    print("|-------------|---------|---------------|----------|----------|")
    print(f"| TF-IDF + LR | {total_classes:7d} | {len(X_train):13d} | {micro_f1_lr:.4f}   | {macro_f1_lr:.4f}   |")
    print(f"| TF-IDF + SVM| {total_classes:7d} | {len(X_train):13d} | {micro_f1_svm:.4f}   | {macro_f1_svm:.4f}   |")
    print()
    print("Target: > 60% Micro F1 for BioBERT retrain on Colab.")
    print(f"Baselines: LR={micro_f1_lr*100:.1f}%, SVM={micro_f1_svm*100:.1f}%")

    # Save filtered dataset for Colab
    out_csv = processed_dir / "mtsamples_classification_filtered.csv"
    df_filtered.to_csv(out_csv, index=False)
    print(f"\nFiltered dataset saved to {out_csv} ({len(df_filtered)} samples, {total_classes} classes)")


if __name__ == "__main__":
    main()
