from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[2]
TRANSFORMER_DIR = BASE_DIR / "outputs" / "models" / "distilbert_sentiment"
TRANSFORMER_METRICS_PATH = BASE_DIR / "outputs" / "metrics" / "distilbert_metrics.json"


class DistilBertSentimentService:
    def __init__(self, model_dir: Path = TRANSFORMER_DIR) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"DistilBERT model not found at {model_dir}. Run python src/train_transformer.py first."
            )

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.labels = [self.model.config.id2label[index] for index in sorted(self.model.config.id2label)]

    def predict_batch(self, texts: list[str]) -> list[dict[str, object]]:
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        predictions: list[dict[str, object]] = []
        for text, probs in zip(texts, probabilities, strict=True):
            probability_map = {
                label: round(float(score), 4) for label, score in zip(self.labels, probs, strict=True)
            }
            predicted_label = max(probability_map, key=probability_map.get)
            predictions.append(
                {
                    "text": text,
                    "predicted_sentiment": predicted_label,
                    "confidence": round(probability_map[predicted_label], 4),
                    "probabilities": probability_map,
                }
            )
        return predictions


def load_transformer_metrics() -> dict:
    if not TRANSFORMER_METRICS_PATH.exists():
        return {}
    return json.loads(TRANSFORMER_METRICS_PATH.read_text())
