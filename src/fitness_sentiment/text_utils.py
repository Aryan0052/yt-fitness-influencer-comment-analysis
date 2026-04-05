from __future__ import annotations

import math
import re
from collections import Counter


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "is", "it", "this", "that",
    "was", "are", "be", "on", "in", "my", "your", "with", "but", "so", "me", "i",
    "we", "you", "us", "at", "too", "very", "just", "one", "from", "how", "after",
    "more", "than", "up", "out", "into", "they", "their", "our", "have", "has",
    "had", "all", "can", "could", "would", "should", "will", "about", "what",
    "when", "why", "who", "which", "really", "been", "am", "im", "it's", "ive",
}


def normalize_text(text: str) -> str:
    cleaned = str(text).strip().lower()
    cleaned = re.sub(r"http\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token]


def get_top_terms(texts: list[str], top_n: int = 40) -> list[dict[str, float]]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in tokenize(text):
            if token in STOPWORDS or len(token) < 3:
                continue
            counter[token] += 1

    if not counter:
        return []

    most_common = counter.most_common(top_n)
    max_count = most_common[0][1]
    results: list[dict[str, float]] = []
    for token, count in most_common:
        size = 14 + (26 * (count / max_count))
        results.append(
            {
                "term": token,
                "count": count,
                "weight": round(math.sqrt(count / max_count), 4),
                "font_size": round(size, 2),
            }
        )
    return results

