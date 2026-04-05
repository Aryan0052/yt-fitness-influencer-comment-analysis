from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.fitness_sentiment.modeling import (
    BEST_MODEL_PATH,
    BEST_MODEL_INFO_PATH,
    DATA_PATH,
    LEADERBOARD_PATH,
    load_best_model,
)
from src.fitness_sentiment.text_utils import get_top_terms
from src.fitness_sentiment.youtube_api import fetch_video_comments, fetch_video_metadata


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Fitness YouTube Sentiment API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    comments: list[str] = Field(min_length=1)


class AnalyzeVideoRequest(BaseModel):
    api_key: str = Field(min_length=10)
    video_id: str = Field(min_length=5)
    max_comments: int = Field(default=100, ge=10, le=200)


def load_leaderboard() -> list[dict[str, Any]]:
    if not LEADERBOARD_PATH.exists():
        return []
    return pd.read_csv(LEADERBOARD_PATH).to_dict(orient="records")


def load_best_model_info() -> dict[str, Any]:
    if not BEST_MODEL_INFO_PATH.exists():
        return {}
    return json.loads(BEST_MODEL_INFO_PATH.read_text())


def load_inference_model():
    return load_best_model()


def summarize_predictions(comments: list[dict[str, Any]]) -> dict[str, Any]:
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for comment in comments:
        sentiment_counts[comment["predicted_sentiment"]] += 1

    total = len(comments) or 1
    sentiment_percentages = {
        label: round((count / total) * 100, 2) for label, count in sentiment_counts.items()
    }

    term_groups = {
        "overall": get_top_terms([comment["text"] for comment in comments], top_n=35),
        "positive": get_top_terms(
            [comment["text"] for comment in comments if comment["predicted_sentiment"] == "positive"],
            top_n=20,
        ),
        "neutral": get_top_terms(
            [comment["text"] for comment in comments if comment["predicted_sentiment"] == "neutral"],
            top_n=20,
        ),
        "negative": get_top_terms(
            [comment["text"] for comment in comments if comment["predicted_sentiment"] == "negative"],
            top_n=20,
        ),
    }

    return {
        "sentiment_counts": sentiment_counts,
        "sentiment_percentages": sentiment_percentages,
        "top_terms": term_groups,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "dataset_ready": DATA_PATH.exists(),
        "model_ready": BEST_MODEL_PATH.exists(),
        "active_model": "SGD Classifier (Calibrated, Optuna Tuned)",
    }


@app.get("/model/summary")
def model_summary() -> dict[str, Any]:
    return {
        "best_model": load_best_model_info(),
        "leaderboard": load_leaderboard(),
        "active_model": "SGD Classifier (Calibrated, Optuna Tuned)",
    }


@app.post("/predict")
def predict_comments(payload: PredictRequest) -> dict[str, Any]:
    try:
        model = load_inference_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probabilities = model.predict_proba(payload.comments)
    labels = model.named_steps["classifier"].classes_
    predictions = model.predict(payload.comments)

    rows = []
    for text, predicted_label, probs in zip(payload.comments, predictions, probabilities, strict=True):
        probability_map = {
            label: round(float(score), 4) for label, score in zip(labels, probs, strict=True)
        }
        rows.append(
            {
                "text": text,
                "predicted_sentiment": predicted_label,
                "confidence": round(max(probability_map.values()), 4),
                "probabilities": probability_map,
            }
        )

    return {
        "results": rows,
        **summarize_predictions(rows),
        "active_model": "SGD Classifier (Calibrated, Optuna Tuned)",
    }


@app.post("/youtube/analyze")
def analyze_youtube_video(payload: AnalyzeVideoRequest) -> dict[str, Any]:
    try:
        model = load_inference_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        video = fetch_video_metadata(payload.video_id, payload.api_key)
        comments = fetch_video_comments(payload.video_id, payload.api_key, payload.max_comments)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not comments:
        raise HTTPException(status_code=404, detail="No comments were returned for this video.")

    comment_texts = [comment["text"] for comment in comments]
    probabilities = model.predict_proba(comment_texts)
    labels = model.named_steps["classifier"].classes_
    predictions = model.predict(comment_texts)

    enriched_comments = []
    for comment, predicted_label, probs in zip(comments, predictions, probabilities, strict=True):
        probability_map = {
            label: round(float(score), 4) for label, score in zip(labels, probs, strict=True)
        }
        enriched_comments.append(
            {
                **comment,
                "predicted_sentiment": predicted_label,
                "confidence": round(max(probability_map.values()), 4),
                "probabilities": probability_map,
            }
        )

    summary = summarize_predictions(enriched_comments)
    return {
        "video": video,
        "total_comments_analyzed": len(enriched_comments),
        "comments": enriched_comments,
        "active_model": "SGD Classifier (Calibrated, Optuna Tuned)",
        **summary,
    }
