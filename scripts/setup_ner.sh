#!/usr/bin/env bash
# Install spacy, scispacy, and the BC5CDR NER model for entity extraction.
# Run from project root with venv activated: bash scripts/setup_ner.sh

set -e
echo "Installing spacy and scispacy..."
pip install spacy scispacy
echo "Installing en_ner_bc5cdr_md model..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
echo "Verifying..."
python -c "import spacy; nlp = spacy.load('en_ner_bc5cdr_md'); print('NER model OK')"
echo "Done."
