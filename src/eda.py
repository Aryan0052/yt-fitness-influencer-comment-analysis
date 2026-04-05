from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_youtube_fitness_comments.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "is", "it", "this", "that",
    "was", "are", "be", "on", "in", "my", "your", "with", "but", "so", "me", "i",
    "we", "you", "us", "at", "too", "very", "just", "one", "from", "how", "after",
    "more", "than", "up", "out", "into", "they", "their", "our", "have", "has",
}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run src/data_prep.py first."
        )
    return pd.read_csv(path)


def save_plot(filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def get_top_words(series: pd.Series, top_n: int = 12) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for text in series.dropna():
        words = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
        counter.update(words)
    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = load_dataset(DATA_PATH)

    plt.figure(figsize=(7, 4))
    sns.countplot(
        data=df,
        x="sentiment",
        order=["positive", "neutral", "negative"],
        hue="sentiment",
        palette="Set2",
        legend=False,
    )
    plt.title("Comment Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Comment Count")
    save_plot("sentiment_distribution.png")

    sentiment_by_influencer = pd.crosstab(df["influencer"], df["sentiment"])
    sentiment_by_influencer = sentiment_by_influencer[[c for c in ["positive", "neutral", "negative"] if c in sentiment_by_influencer.columns]]
    sentiment_by_influencer.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="viridis")
    plt.title("Sentiment by Influencer")
    plt.xlabel("Influencer")
    plt.ylabel("Comment Count")
    save_plot("sentiment_by_influencer.png")

    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=df,
        x="sentiment",
        y="engagement_score",
        order=["positive", "neutral", "negative"],
        hue="sentiment",
        palette="magma",
        estimator="mean",
        errorbar=None,
        legend=False,
    )
    plt.title("Average Engagement Score by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Engagement Score")
    save_plot("engagement_by_sentiment.png")

    plt.figure(figsize=(8, 4))
    sns.boxplot(
        data=df,
        x="sentiment",
        y="word_count",
        order=["positive", "neutral", "negative"],
        hue="sentiment",
        palette="pastel",
        legend=False,
    )
    plt.title("Comment Length by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Word Count")
    save_plot("comment_length_by_sentiment.png")

    top_positive = get_top_words(df.loc[df["sentiment"] == "positive", "clean_comment"])
    plt.figure(figsize=(8, 4))
    sns.barplot(data=top_positive, x="count", y="word", hue="word", dodge=False, palette="Greens_r", legend=False)
    plt.title("Most Common Positive Words")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    save_plot("top_positive_words.png")

    top_negative = get_top_words(df.loc[df["sentiment"] == "negative", "clean_comment"])
    plt.figure(figsize=(8, 4))
    sns.barplot(data=top_negative, x="count", y="word", hue="word", dodge=False, palette="Reds_r", legend=False)
    plt.title("Most Common Negative Words")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    save_plot("top_negative_words.png")

    summary = (
        df.groupby("influencer", as_index=False)
        .agg(
            comments=("comment_text", "count"),
            avg_likes=("like_count", "mean"),
            avg_replies=("reply_count", "mean"),
            avg_engagement=("engagement_score", "mean"),
        )
        .round(2)
    )
    summary.to_csv(FIGURES_DIR / "influencer_summary.csv", index=False)

    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
