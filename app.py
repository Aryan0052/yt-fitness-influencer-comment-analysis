import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.fitness_sentiment.modeling import BEST_MODEL_INFO_PATH, BEST_MODEL_PATH, DATA_PATH, LEADERBOARD_PATH, load_best_model


BASE_DIR = Path(__file__).resolve().parent
SENTIMENT_COLORS = {"positive": "#1b7f5b", "neutral": "#a67c00", "negative": "#b33a3a"}


def load_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def load_best_metrics() -> dict:
    if not BEST_MODEL_INFO_PATH.exists():
        return {}
    return json.loads(BEST_MODEL_INFO_PATH.read_text())


def load_leaderboard() -> pd.DataFrame:
    if not LEADERBOARD_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LEADERBOARD_PATH)


def format_percent(value: float) -> str:
    return f"{value:.1%}"


st.set_page_config(
    page_title="FitScope Comments",
    page_icon="FS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f7f1e8;
            --card: rgba(255, 252, 246, 0.88);
            --text: #1f2a37;
            --muted: #5f6b7a;
            --line: rgba(48, 65, 84, 0.12);
            --accent: #d45b2c;
            --accent-soft: rgba(212, 91, 44, 0.12);
            --teal: #1b7f5b;
            --gold: #a67c00;
            --rose: #b33a3a;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(212, 91, 44, 0.18), transparent 24%),
                radial-gradient(circle at top right, rgba(27, 127, 91, 0.16), transparent 22%),
                linear-gradient(180deg, #fffaf1 0%, var(--bg) 55%, #efe4d5 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"], [data-testid="stHeader"] {
            display: none;
        }

        .hero {
            border-radius: 28px;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,247,239,0.76));
            border: 1px solid rgba(255,255,255,0.7);
            box-shadow: 0 20px 60px rgba(92, 64, 35, 0.12);
            margin-bottom: 1rem;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }

        .hero h1 {
            font-size: clamp(2.5rem, 5vw, 4.5rem);
            line-height: 0.96;
            margin-bottom: 0.8rem;
            letter-spacing: -0.04em;
        }

        .hero p {
            max-width: 760px;
            color: var(--muted);
            line-height: 1.75;
            font-size: 1rem;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.2rem;
        }

        .stat-card, .panel {
            background: var(--card);
            border: 1px solid rgba(255,255,255,0.72);
            border-radius: 22px;
            box-shadow: 0 16px 40px rgba(99, 70, 42, 0.08);
            padding: 1.1rem;
        }

        .label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.76rem;
        }

        .value {
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .result-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            color: white;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }

        @media (max-width: 900px) {
            .stat-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

dataset = load_dataset()
leaderboard = load_leaderboard()
metrics = load_best_metrics()

try:
    model = load_best_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">FitScope Comments | NLP + Chrome Extension</div>
        <h1>Train, compare, and serve sentiment models for YouTube fitness comments.</h1>
        <p>
            Use this Streamlit page as the local project console. It summarizes the training dataset,
            shows the winning model from the 9-model leaderboard, and lets you test the same sentiment
            classifier that powers the Chrome extension and FastAPI backend.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

if dataset is None:
    st.warning("Processed dataset not found. Run `python src/data_prep.py` first.")
    st.stop()

positive_rate = (dataset["sentiment"] == "positive").mean()
neutral_rate = (dataset["sentiment"] == "neutral").mean()
avg_engagement = dataset["engagement_score"].mean()
top_influencer = (
    dataset.groupby("influencer")["like_count"].mean().sort_values(ascending=False).index[0]
)

st.markdown(
    f"""
    <div class="stat-grid">
        <div class="stat-card"><div class="label">Comments</div><div class="value">{len(dataset)}</div></div>
        <div class="stat-card"><div class="label">Positive Share</div><div class="value">{format_percent(positive_rate)}</div></div>
        <div class="stat-card"><div class="label">Neutral Share</div><div class="value">{format_percent(neutral_rate)}</div></div>
        <div class="stat-card"><div class="label">Top Avg Likes</div><div class="value">{top_influencer}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if metrics:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", format_percent(metrics.get("accuracy", 0)))
    metric_cols[1].metric("Macro Precision", format_percent(metrics.get("precision_macro", 0)))
    metric_cols[2].metric("Macro Recall", format_percent(metrics.get("recall_macro", 0)))
    metric_cols[3].metric("Macro F1", format_percent(metrics.get("f1_macro", 0)))

summary_tab, predictor_tab, leaderboard_tab = st.tabs(["Dataset Dashboard", "Predict Comment Sentiment", "Model Leaderboard"])

with summary_tab:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Sentiment Mix by Influencer")
        sentiment_mix = pd.crosstab(dataset["influencer"], dataset["sentiment"])
        sentiment_mix = sentiment_mix.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
        st.bar_chart(sentiment_mix)

        st.subheader("Average Engagement by Sentiment")
        engagement_view = dataset.groupby("sentiment", as_index=False)["engagement_score"].mean()
        engagement_view = engagement_view.set_index("sentiment").reindex(["positive", "neutral", "negative"])
        st.bar_chart(engagement_view)

    with right:
        st.subheader("Influencer Snapshot")
        influencer_summary = (
            dataset.groupby("influencer", as_index=False)
            .agg(
                comments=("comment_text", "count"),
                avg_likes=("like_count", "mean"),
                avg_replies=("reply_count", "mean"),
                avg_engagement=("engagement_score", "mean"),
            )
            .sort_values("avg_engagement", ascending=False)
            .round(2)
        )
        st.dataframe(influencer_summary, use_container_width=True, hide_index=True)

        st.subheader("Recent Comment Sample")
        st.dataframe(
            dataset[["influencer", "video_title", "comment_text", "sentiment", "engagement_score"]].head(12),
            use_container_width=True,
            hide_index=True,
        )

with predictor_tab:
    st.subheader("Classify a New YouTube Comment")
    sample_text = st.text_area(
        "Paste a comment",
        value="This was one of the clearest workout explanations I have seen all month.",
        height=140,
    )

    if st.button("Predict Sentiment"):
        predicted_label = model.predict([sample_text])[0]
        probabilities = model.predict_proba([sample_text])[0]
        probability_df = pd.DataFrame(
            {
                "sentiment": model.named_steps["classifier"].classes_,
                "probability": probabilities,
            }
        ).sort_values("probability", ascending=False)

        pill_color = SENTIMENT_COLORS.get(predicted_label, "#334155")
        st.markdown(
            f"<div class='result-pill' style='background:{pill_color};'>{predicted_label.title()}</div>",
            unsafe_allow_html=True,
        )
        st.write(f"Top prediction confidence: **{format_percent(float(probability_df.iloc[0]['probability']))}**")
        st.bar_chart(probability_df.set_index("sentiment"))

        if predicted_label == "positive":
            st.success("This looks like supportive or appreciative audience feedback.")
        elif predicted_label == "negative":
            st.error("This looks like criticism, frustration, or dissatisfaction.")
        else:
            st.info("This reads more like a request, observation, or balanced reaction.")

with leaderboard_tab:
    st.subheader("Best Model")
    if metrics:
        st.json(metrics)
    else:
        st.info("Run `python src/train_model.py` to generate the 9-model comparison and save the best model.")

    st.subheader("All Model Results")
    if leaderboard.empty:
        st.warning("Leaderboard not found yet. Run `python src/train_model.py` first.")
    else:
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)
        st.bar_chart(leaderboard.set_index("model")[["f1_macro", "accuracy"]])

st.caption(f"Average engagement score across the dataset: {avg_engagement:.1f}")
