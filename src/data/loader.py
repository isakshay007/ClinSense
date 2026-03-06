"""MTSamples data loading and preprocessing for ClinSense."""

from pathlib import Path
from typing import Optional

import pandas as pd


def download_mtsamples_kaggle(
    raw_dir: Path,
    dataset: str = "louiscia/transcription-samples-mtsamples",
) -> Path:
    """
    Download MTSamples dataset from Kaggle.
    
    Requires kaggle.json in ~/.kaggle/ or KAGGLE_USERNAME/KAGGLE_KEY env vars.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            dataset,
            path=str(raw_dir),
            unzip=True,
        )
        return raw_dir
    except Exception as e:
        raise RuntimeError(
            f"Kaggle download failed. Ensure kaggle.json is in ~/.kaggle/ or set "
            f"KAGGLE_USERNAME/KAGGLE_KEY. Error: {e}"
        ) from e


def load_mtsamples_from_csv(
    data_path: Path,
) -> pd.DataFrame:
    """
    Load pre-filtered MTSamples from CSV (e.g. mtsamples_classification_filtered.csv).
    No filtering applied; use for optimal-filtered datasets.
    """
    df = pd.read_csv(data_path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    col_map = {"Unnamed: 0": "note_id", "0": "note_id"}
    df = df.rename(columns=col_map)
    if "medical_specialty" not in df.columns and "specialty" in df.columns:
        df = df.rename(columns={"specialty": "medical_specialty"})
    text_col = "transcription" if "transcription" in df.columns else "text"
    df["transcription"] = df[text_col].fillna("").astype(str).str.strip()
    if "medical_specialty" not in df.columns:
        raise ValueError("CSV must have medical_specialty or specialty column")
    df["medical_specialty"] = df["medical_specialty"].fillna("").astype(str).str.strip()
    df = df[df["transcription"].str.len() >= 50].copy()
    return df.reset_index(drop=True)


def load_mtsamples(
    data_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    download_if_missing: bool = True,
    dataset: str = "louiscia/transcription-samples-mtsamples",
    min_samples_per_class: int = 2,
    top_n_classes: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load MTSamples dataset from CSV.
    
    Supports multiple possible file locations and column naming conventions.
    """
    if data_path is None and raw_dir is not None:
        data_path = raw_dir / "mtsamples.csv"
    
    if data_path is None:
        raise ValueError("Either data_path or raw_dir must be provided")
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        if download_if_missing and raw_dir is not None:
            download_mtsamples_kaggle(raw_dir, dataset)
            data_path = raw_dir / "mtsamples.csv"
        else:
            raise FileNotFoundError(
                f"MTSamples not found at {data_path}. Run with download_if_missing=True "
                "and configure Kaggle credentials."
            )
    
    # Handle CSV with possible unnamed first column (index)
    df = pd.read_csv(
        data_path,
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False,
    )
    
    # Standardize column names (some versions have different naming)
    col_map = {
        "Unnamed: 0": "note_id",
        "0": "note_id",
    }
    df = df.rename(columns=col_map)
    
    # Ensure required columns exist
    required = {"transcription", "medical_specialty"}
    if "medical_specialty" not in df.columns:
        if "medical_specialty" in df.columns.str.lower():
            df = df.rename(columns={c: "medical_specialty" for c in df.columns if c.lower() == "medical_specialty"})
    
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Found: {list(df.columns)}")
    
    # Clean text fields
    df["transcription"] = df["transcription"].fillna("").astype(str).str.strip()
    df["medical_specialty"] = df["medical_specialty"].fillna("").astype(str).str.strip()
    
    # Filter empty or too-short transcriptions (min 50 chars)
    df = df[df["transcription"].str.len() >= 50].copy()
    
    # Filter classes: min_samples_per_class, optionally top_n by count
    specialty_counts = df["medical_specialty"].value_counts()
    valid = specialty_counts[specialty_counts >= min_samples_per_class]
    if top_n_classes is not None:
        valid = valid.head(top_n_classes)
    valid_specialties = valid.index
    df = df[df["medical_specialty"].isin(valid_specialties)].copy()
    
    df = df.reset_index(drop=True)
    return df


def prepare_classification_data(
    df: pd.DataFrame,
    text_col: str = "transcription",
    label_col: str = "medical_specialty",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, list[str]]:
    """
    Split data for multi-class classification with stratified splits.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, label_names
    """
    from sklearn.model_selection import train_test_split
    
    X = df[text_col]
    y = df[label_col]
    
    # Stratified split: first train+val vs test, then train vs val
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    label_names = sorted(y.unique().tolist())
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_names
