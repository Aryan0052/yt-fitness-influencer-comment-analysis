from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "youtube_fitness_comments.csv"

INFLUENCER_VIDEOS = {
    "FitWithMaya": [
        "12 Minute Glute Burner",
        "Low Impact Morning Workout",
        "Healthy Meal Prep for Fat Loss",
        "Beginner Friendly Full Body Session",
    ],
    "CoachRavi": [
        "Push Pull Legs Explained",
        "Affordable High Protein Indian Meals",
        "How To Improve Squat Form",
        "Resistance Band Home Workout",
    ],
    "IronLex": [
        "Science Based Muscle Gain Tips",
        "Upper Body Strength Session",
        "Cutting Diet Mistakes",
        "Hypertrophy Training Fundamentals",
    ],
    "JeetSelal": [
        "Indian Bulking Diet Reality",
        "Best Workout Split for Natural Lifters",
        "How To Stay Motivated in the Gym",
        "Fat Loss Cardio vs Weights",
    ],
    "ShreddedNisha": [
        "Women's Strength Training Basics",
        "Home Abs and Mobility Flow",
        "What I Eat Before and After Workouts",
        "Common Beginner Mistakes in Fitness",
    ],
}

POSITIVE_TEMPLATES = [
    "This {topic} video was exactly what I needed today.",
    "Your explanation made {topic} so much easier to understand.",
    "One of the best fitness uploads on this channel so far.",
    "I tried this routine and the results already feel promising.",
    "Clear coaching, great pacing, and very motivating energy.",
    "This helped me stay consistent with my training this week.",
    "I appreciate how realistic and practical your advice is.",
    "Amazing content, especially the way you explained form cues.",
    "This is now part of my regular workout rotation.",
    "Finally a fitness video that feels useful instead of confusing.",
]

NEUTRAL_TEMPLATES = [
    "Good video overall, but I would like a bit more detail on {topic}.",
    "This was helpful and straightforward for a general audience.",
    "Nice explanation, though a beginner version would also help.",
    "The content is solid, but timestamps would make it easier to follow.",
    "Useful video and I saved it for later reference.",
    "I liked the structure, but a few more examples would be nice.",
    "This seems practical, especially for people training at home.",
    "Could you make a follow up focused only on {topic} next time?",
    "The advice makes sense, though I still have a few questions.",
    "Decent upload and the information is easy to apply.",
]

NEGATIVE_TEMPLATES = [
    "The advice on {topic} felt a bit rushed and incomplete.",
    "I expected more practical examples instead of general tips.",
    "The camera angle made the exercise demonstration hard to follow.",
    "This one felt repetitive compared to your better videos.",
    "I disagree with a few points in the {topic} section.",
    "Too much promotion and not enough actual coaching in this upload.",
    "The pacing was off and the key points were easy to miss.",
    "I wish this had more safety guidance for beginners.",
    "The editing cuts made it harder to stay focused on the explanation.",
    "This topic deserved a clearer and more detailed breakdown.",
]

TOPICS = [
    "workout form",
    "meal prep",
    "progressive overload",
    "fat loss",
    "bulking",
    "mobility",
    "home training",
    "protein intake",
    "gym motivation",
    "exercise technique",
]

TAIL_PHRASES = [
    "I am saving this for later.",
    "This stood out more than I expected.",
    "I would watch a follow up on this.",
    "This feels relevant for natural lifters too.",
    "The practical examples really matter here.",
    "This would help a lot of beginners.",
    "The pacing made the point easier to catch.",
    "I noticed this immediately while watching.",
    "This is useful for people training at home.",
    "I hope you explore this topic again.",
]


def build_generated_rows(target_count_per_class: int = 72) -> pd.DataFrame:
    source = pd.read_csv(RAW_DATA_PATH)
    generated_rows: list[dict[str, object]] = []
    class_counts = source["sentiment"].value_counts().to_dict()

    template_map = {
        "positive": POSITIVE_TEMPLATES,
        "neutral": NEUTRAL_TEMPLATES,
        "negative": NEGATIVE_TEMPLATES,
    }

    influencers = list(INFLUENCER_VIDEOS.keys())
    for sentiment, templates in template_map.items():
        current_count = class_counts.get(sentiment, 0)
        needed = max(0, target_count_per_class - current_count)
        for index in range(needed):
            influencer = influencers[(index * 3) % len(influencers)]
            video_title = INFLUENCER_VIDEOS[influencer][(index * 5) % len(INFLUENCER_VIDEOS[influencer])]
            template = templates[(index * 7) % len(templates)]
            topic = TOPICS[(index * 11) % len(TOPICS)]
            tail = TAIL_PHRASES[(index * 13) % len(TAIL_PHRASES)]

            like_base = {"positive": 120, "neutral": 55, "negative": 28}[sentiment]
            reply_base = {"positive": 6, "neutral": 3, "negative": 4}[sentiment]

            generated_rows.append(
                {
                    "influencer": influencer,
                    "video_title": video_title,
                    "comment_text": f"{template.format(topic=topic)} {tail}",
                    "like_count": like_base + (index % 35),
                    "reply_count": reply_base + (index % 5),
                    "days_since_post": 1 + (index % 14),
                    "sentiment": sentiment,
                }
            )

    generated_df = pd.DataFrame(generated_rows)
    combined = pd.concat([source, generated_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["influencer", "video_title", "comment_text"]).reset_index(drop=True)
    return combined


def main() -> None:
    expanded = build_generated_rows()
    expanded.to_csv(RAW_DATA_PATH, index=False)
    print("Expanded dataset saved to:", RAW_DATA_PATH)
    print(expanded["sentiment"].value_counts().to_string())
    print("Rows:", len(expanded))


if __name__ == "__main__":
    main()
