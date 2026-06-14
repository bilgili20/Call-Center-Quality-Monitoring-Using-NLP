from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "berturk_call_quality_regression"
TOKENIZER_FALLBACK = "dbmdz/bert-base-turkish-cased"


@lru_cache(maxsize=1)
def load_quality_model():
    """Load the tokenizer and fine-tuned BERTurk regression model."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model klasörü bulunamadı: {MODEL_DIR}. "
            "Model dosyalarını models/berturk_call_quality_regression altına yerleştirin."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FALLBACK, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_quality_score(text, tokenizer, model, device, max_length=256):
    """
    Predict a 0-100 call quality score from a Turkish call center transcript.

    The model output is a normalized regression value, so inference multiplies
    it by 100 and clips it to the valid score range.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Transcript metni boş olamaz.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        raw_score = outputs.logits.squeeze().item()

    score = float(np.clip(raw_score * 100, 0, 100))
    return round(score, 2)


def score_to_class(score):
    """Convert a numeric score to the reporting quality class."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Average"
    return "Poor"


def manual_review_required(score):
    """Return True when the call should be reviewed manually."""
    return score < 60


def generate_basic_comment(score):
    """Generate a short Turkish UI comment for the predicted score."""
    quality_class = score_to_class(score)
    comments = {
        "Excellent": "The representative appears to have handled the call at a very high quality level.",
        "Good": "The call is generally good quality; there may be minor areas for improvement.",
        "Average": "The call is average; it may be reviewed for consistency, clarity or service quality.",
        "Poor": "The call should be reviewed carefully for quality and coaching needs.",
    }
    return comments[quality_class]


# Backward-compatible alias for older local imports.
load_model = load_quality_model
