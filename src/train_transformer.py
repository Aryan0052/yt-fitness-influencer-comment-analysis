from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_youtube_fitness_comments.csv"
TRANSFORMER_DIR = BASE_DIR / "outputs" / "models" / "distilbert_sentiment"
TRANSFORMER_METRICS_PATH = BASE_DIR / "outputs" / "metrics" / "distilbert_metrics.json"
TRANSFORMER_REPORT_PATH = BASE_DIR / "outputs" / "metrics" / "distilbert_classification_report.txt"
MODEL_NAME = "distilbert-base-uncased"


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


class CommentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=128,
        )
        encoded["labels"] = self.labels[idx]
        return encoded


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at {DATA_PATH}. Run python src/data_prep.py first.")
    return pd.read_csv(DATA_PATH)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def main() -> None:
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_comment"],
        df["sentiment"].map(LABEL_TO_ID),
        test_size=0.25,
        random_state=42,
        stratify=df["sentiment"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = CommentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_dataset = CommentDataset(X_test.tolist(), y_test.tolist(), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / "outputs" / "tmp_distilbert"),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)

    metrics = {
        "model": "DistilBERT",
        "accuracy": round(accuracy_score(y_test, predicted_labels), 4),
        "precision_macro": round(precision_score(y_test, predicted_labels, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, predicted_labels, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, predicted_labels, average="macro", zero_division=0), 4),
    }

    TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMER_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(TRANSFORMER_DIR))
    tokenizer.save_pretrained(str(TRANSFORMER_DIR))
    TRANSFORMER_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    TRANSFORMER_REPORT_PATH.write_text(
        classification_report(
            y_test,
            predicted_labels,
            target_names=[ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL.keys())],
            zero_division=0,
        )
    )

    print("Transformer metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved DistilBERT model to: {TRANSFORMER_DIR}")
    print(f"Saved metrics to: {TRANSFORMER_METRICS_PATH}")


if __name__ == "__main__":
    main()
