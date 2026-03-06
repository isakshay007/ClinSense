"""
scispaCy-based NER for extracting drugs, diseases, and symptoms from clinical text.

Uses en_ner_bc5cdr_md for chemicals/drugs and diseases.
For symptoms, we rely on disease entities and keyword expansion.
"""

from typing import Optional

import pandas as pd


def _load_scispacy_model(model_name: str = "en_ner_bc5cdr_md"):
    """Lazy load scispaCy NER model."""
    try:
        import spacy
        return spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"scispaCy model '{model_name}' not found. Install with:\n"
            f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
            f"v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
        )


_nlp = None


def get_nlp(model_name: str = "en_ner_bc5cdr_md"):
    """Get or load the scispaCy NLP model."""
    global _nlp
    if _nlp is None:
        _nlp = _load_scispacy_model(model_name)
    return _nlp


def extract_entities_scispacy(
    text: str,
    model_name: str = "en_ner_bc5cdr_md",
) -> list[dict]:
    """
    Extract medical entities (drugs, diseases) from clinical text.
    
    BC5CDR model recognizes:
    - CHEMICAL (drugs, compounds)
    - DISEASE
    
    Returns list of {"text": str, "label": str, "start": int, "end": int}
    """
    nlp = get_nlp(model_name)
    doc = nlp(text[:1000000])  # Limit length to avoid OOM
    
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


def extract_entities_batch(
    texts: list[str],
    model_name: str = "en_ner_bc5cdr_md",
    n_process: int = 1,
) -> list[list[dict]]:
    """Extract entities from a batch of texts."""
    nlp = get_nlp(model_name)
    docs = list(nlp.pipe(texts, n_process=n_process))
    
    results = []
    for doc in docs:
        entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
        results.append(entities)
    return results


def add_entity_columns(df: pd.DataFrame, text_col: str = "transcription") -> pd.DataFrame:
    """
    Add extracted entity columns to DataFrame.
    
    Adds: entities_raw (list of dicts), drugs, diseases, chemicals.
    """
    results = extract_entities_batch(df[text_col].fillna("").astype(str).tolist())
    
    def flatten_labels(ents: list[dict], label: str) -> list[str]:
        return [e["text"] for e in ents if e["label"] == label]
    
    df = df.copy()
    df["entities_raw"] = results
    df["drugs"] = [flatten_labels(r, "CHEMICAL") for r in results]
    df["diseases"] = [flatten_labels(r, "DISEASE") for r in results]
    df["chemicals"] = [flatten_labels(r, "CHEMICAL") for r in results]
    return df
