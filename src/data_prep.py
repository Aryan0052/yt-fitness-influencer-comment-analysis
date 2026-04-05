from pathlib import Path

import pandas as pd

from fitness_sentiment.text_utils import normalize_text


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "youtube_fitness_comments.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_youtube_fitness_comments.csv"

VALID_SENTIMENTS = {"positive", "neutral", "negative"}


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the raw comments CSV in data/raw/."
        )
    return pd.read_csv(path)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower() for col in cleaned.columns]

    required_columns = {
        "influencer",
        "video_title",
        "comment_text",
        "like_count",
        "reply_count",
        "days_since_post",
        "sentiment",
    }
    missing = required_columns.difference(cleaned.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = cleaned.dropna(subset=["comment_text", "sentiment", "influencer", "video_title"])
    cleaned["sentiment"] = cleaned["sentiment"].str.strip().str.lower()
    cleaned = cleaned[cleaned["sentiment"].isin(VALID_SENTIMENTS)].copy()

    numeric_columns = ["like_count", "reply_count", "days_since_post"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=numeric_columns)
    cleaned = cleaned.drop_duplicates(subset=["influencer", "video_title", "comment_text"])

    cleaned["clean_comment"] = cleaned["comment_text"].map(normalize_text)
    cleaned = cleaned[cleaned["clean_comment"].str.len() > 0].copy()

    cleaned["comment_length"] = cleaned["clean_comment"].str.len()
    cleaned["word_count"] = cleaned["clean_comment"].str.split().str.len()
    cleaned["engagement_score"] = cleaned["like_count"] + (2 * cleaned["reply_count"])
    cleaned["has_question"] = cleaned["comment_text"].str.contains(r"\?", regex=True).astype(int)
    cleaned["has_suggestion"] = cleaned["clean_comment"].str.contains(
        r"\\b(?:can|could|please|wish)\\b", regex=True
    ).astype(int)

    return cleaned.sort_values(["influencer", "video_title"]).reset_index(drop=True)


def main() -> None:
    df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_data(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Raw shape:", df.shape)
    print("Cleaned shape:", cleaned_df.shape)
    print("Sentiment distribution:")
    print(cleaned_df["sentiment"].value_counts().to_string())
    print(f"Saved cleaned data to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()

