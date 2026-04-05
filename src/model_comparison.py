from fitness_sentiment.modeling import LEADERBOARD_PATH, train_and_select_best


def main() -> None:
    leaderboard, _, _, _ = train_and_select_best()
    print(leaderboard.to_string(index=False))
    print("Best-performing model is shown at the top of the leaderboard.")
    print("Run `python src/train_model.py` to save the winning model and metadata.")
    print(f"Leaderboard output path: {LEADERBOARD_PATH}")


if __name__ == "__main__":
    main()
