import json

from fitness_sentiment.modeling import (
    BEST_MODEL_INFO_PATH,
    BEST_MODEL_PATH,
    CLASSIFICATION_REPORT_PATH,
    LEADERBOARD_PATH,
    TUNING_SUMMARY_PATH,
    TUNING_TRIALS_PATH,
    save_training_outputs,
    train_and_select_best,
)


def main() -> None:
    leaderboard, best_metrics, best_pipeline, report, trials_df = train_and_select_best()
    save_training_outputs(leaderboard, best_metrics, best_pipeline, report, trials_df)

    print("Model leaderboard:")
    print(leaderboard.to_string(index=False))
    print()
    print("Best model summary:")
    print(json.dumps(best_metrics, indent=2))
    print()
    print("Classification report:")
    print(report)
    print(f"Saved best model to: {BEST_MODEL_PATH}")
    print(f"Saved leaderboard to: {LEADERBOARD_PATH}")
    print(f"Saved best model summary to: {BEST_MODEL_INFO_PATH}")
    print(f"Saved classification report to: {CLASSIFICATION_REPORT_PATH}")
    print(f"Saved Optuna summary to: {TUNING_SUMMARY_PATH}")
    print(f"Saved Optuna trials to: {TUNING_TRIALS_PATH}")


if __name__ == "__main__":
    main()
