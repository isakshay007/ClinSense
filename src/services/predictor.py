"""BERT-based medical specialty prediction service."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SpecialtyPredictor:
    """Load BERT model once and predict medical specialty."""

    def __init__(self, model_path: str | Path, max_length: int = 512):
        self.model_path = Path(model_path)
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def load(self) -> "SpecialtyPredictor":
        """Lazy load model and tokenizer."""
        if self._model is not None:
            return self

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run Colab notebook and download clinsense_bert_final to models/"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.eval()
        return self

    def predict(self, text: str) -> str:
        """Predict medical specialty for given text."""
        self.load()

        inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.no_grad():
            logits = self._model(**inputs).logits
        pred_id = logits.argmax(dim=-1).item()

        id2label = self._model.config.id2label
        label = id2label.get(str(pred_id), id2label.get(pred_id, f"class_{pred_id}"))
        return label

    def predict_proba(self, text: str) -> tuple[str, dict[str, float]]:
        """Predict specialty and return label with per-class probabilities."""
        self.load()

        inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()

        id2label = self._model.config.id2label
        pred_id = probs.index(max(probs))
        label = id2label.get(str(pred_id), id2label.get(pred_id, f"class_{pred_id}"))

        prob_dict = {id2label.get(str(i), id2label.get(i, f"class_{i}")): p for i, p in enumerate(probs)}
        return label, prob_dict
