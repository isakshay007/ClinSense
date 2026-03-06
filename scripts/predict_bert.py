#!/usr/bin/env python3
"""
Predict medical specialty using fine-tuned BERT (bert-base-uncased, partial fine-tune).

Usage:
  python scripts/predict_bert.py "Your clinical note text here"
  python scripts/predict_bert.py --file path/to/note.txt
  python scripts/predict_bert.py  # interactive mode
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.predictor import SpecialtyPredictor


def load_model_and_predict(model_path: Path, text: str, max_length: int = 512) -> str:
    """Load BERT and predict medical specialty."""
    predictor = SpecialtyPredictor(model_path, max_length=max_length)
    return predictor.predict(text)


def main():
    parser = argparse.ArgumentParser(description="Predict medical specialty using BERT")
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Clinical note text to classify",
    )
    parser.add_argument(
        "--model",
        default="models/bert_finetuned",
        help="Path to saved BERT model",
    )
    parser.add_argument(
        "--file",
        help="Path to file containing text",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length (default: 512)",
    )
    args = parser.parse_args()

    model_path = Path(__file__).parent.parent / args.model

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        text = path.read_text()
        label = load_model_and_predict(model_path, text, args.max_length)
        print(f"Predicted specialty: {label}")
        return

    if args.text:
        text = args.text
    else:
        print("Enter clinical note (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        text = "\n".join(lines).strip() or "No input provided."

    if not text or text == "No input provided.":
        print("No text provided.")
        sys.exit(1)

    label = load_model_and_predict(model_path, text, args.max_length)
    print(f"Predicted specialty: {label}")


if __name__ == "__main__":
    main()
