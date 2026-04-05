from __future__ import annotations

import json
from pathlib import Path

import joblib
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_youtube_fitness_comments.csv"
MODELS_DIR = BASE_DIR / "outputs" / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
BEST_MODEL_PATH = MODELS_DIR / "best_fitness_sentiment_model.pkl"
BEST_MODEL_INFO_PATH = METRICS_DIR / "best_model_summary.json"
LEADERBOARD_PATH = METRICS_DIR / "model_leaderboard.csv"
CLASSIFICATION_REPORT_PATH = METRICS_DIR / "best_model_classification_report.txt"
TUNING_SUMMARY_PATH = METRICS_DIR / "optuna_tuning_summary.json"
TUNING_TRIALS_PATH = METRICS_DIR / "optuna_trials.csv"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run src/data_prep.py first."
        )
    return pd.read_csv(path)


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )


def build_vectorizer_from_params(params: dict[str, int | float | str]) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, int(params["ngram_upper"])),
        min_df=int(params["min_df"]),
        max_df=float(params["max_df"]),
        sublinear_tf=bool(params["sublinear_tf"]),
        lowercase=True,
    )


def build_model_catalog() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Linear SVC (Calibrated)": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
        "SGD Classifier (Calibrated)": CalibratedClassifierCV(
            SGDClassifier(loss="modified_huber", class_weight="balanced", random_state=42),
            cv=3,
        ),
        "Passive Aggressive (Calibrated)": CalibratedClassifierCV(
            PassiveAggressiveClassifier(class_weight="balanced", random_state=42, max_iter=2000),
            cv=3,
        ),
        "Ridge Classifier (Calibrated)": CalibratedClassifierCV(RidgeClassifier(class_weight="balanced"), cv=3),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Complement Naive Bayes": ComplementNB(),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }


def build_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", build_vectorizer()),
            ("classifier", estimator),
        ]
    )


def build_tuned_sgd_pipeline(params: dict[str, int | float | str]) -> Pipeline:
    estimator = SGDClassifier(
        loss=str(params["loss"]),
        alpha=float(params["alpha"]),
        penalty=str(params["penalty"]),
        max_iter=int(params["max_iter"]),
        tol=float(params["tol"]),
        class_weight="balanced",
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(estimator, cv=3)
    return Pipeline(
        steps=[
            ("vectorizer", build_vectorizer_from_params(params)),
            ("classifier", calibrated),
        ]
    )


def evaluate_pipeline(name: str, pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict[str, float | str]:
    predictions = pipeline.predict(X_test)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision_macro": round(precision_score(y_test, predictions, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, predictions, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, predictions, average="macro", zero_division=0), 4),
    }


def build_train_test_split() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = load_dataset()
    X = df["clean_comment"]
    y = df["sentiment"]
    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )


def tune_best_candidate(
    X_train: pd.Series,
    y_train: pd.Series,
    n_trials: int = 40,
) -> tuple[Pipeline, dict[str, float | str], pd.DataFrame]:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "ngram_upper": trial.suggest_int("ngram_upper", 1, 2),
            "min_df": trial.suggest_int("min_df", 1, 2),
            "max_df": trial.suggest_float("max_df", 0.85, 1.0),
            "sublinear_tf": trial.suggest_categorical("sublinear_tf", [True, False]),
            "loss": trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-3, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "max_iter": trial.suggest_int("max_iter", 1000, 3000, step=500),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
        }
        pipeline = build_tuned_sgd_pipeline(params)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "f1_macro": "f1_macro",
                "recall_macro": "recall_macro",
                "accuracy": "accuracy",
            },
            n_jobs=1,
        )
        mean_f1 = scores["test_f1_macro"].mean()
        mean_recall = scores["test_recall_macro"].mean()
        mean_accuracy = scores["test_accuracy"].mean()
        trial.set_user_attr("f1_macro", round(float(mean_f1), 4))
        trial.set_user_attr("recall_macro", round(float(mean_recall), 4))
        trial.set_user_attr("accuracy", round(float(mean_accuracy), 4))
        return (0.65 * mean_f1) + (0.25 * mean_recall) + (0.10 * mean_accuracy)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    best_pipeline = build_tuned_sgd_pipeline(best_params)
    best_pipeline.fit(X_train, y_train)

    best_summary = {
        "model": "SGD Classifier (Calibrated, Optuna Tuned)",
        "objective_score": round(float(study.best_value), 4),
        "cv_f1_macro": study.best_trial.user_attrs.get("f1_macro"),
        "cv_recall_macro": study.best_trial.user_attrs.get("recall_macro"),
        "cv_accuracy": study.best_trial.user_attrs.get("accuracy"),
        "best_params": best_params,
        "n_trials": n_trials,
    }
    trials_df = study.trials_dataframe()
    return best_pipeline, best_summary, trials_df


def train_and_select_best() -> tuple[pd.DataFrame, dict[str, float | str], Pipeline, str]:
    X_train, X_test, y_train, y_test = build_train_test_split()

    results: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, estimator in build_model_catalog().items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        results.append(evaluate_pipeline(name, pipeline, X_test, y_test))

    leaderboard = pd.DataFrame(results).sort_values(
        by=["f1_macro", "accuracy", "precision_macro"],
        ascending=False,
    ).reset_index(drop=True)

    best_name = str(leaderboard.iloc[0]["model"])
    best_pipeline = fitted_models[best_name]
    best_metrics = leaderboard.iloc[0].to_dict()

    tuned_pipeline, tuning_summary, trials_df = tune_best_candidate(X_train, y_train)
    tuned_metrics = evaluate_pipeline(
        "SGD Classifier (Calibrated, Optuna Tuned)",
        tuned_pipeline,
        X_test,
        y_test,
    )
    leaderboard = pd.concat([leaderboard, pd.DataFrame([tuned_metrics])], ignore_index=True)
    leaderboard = leaderboard.sort_values(
        by=["f1_macro", "recall_macro", "accuracy", "precision_macro"],
        ascending=False,
    ).reset_index(drop=True)

    if str(leaderboard.iloc[0]["model"]) == tuned_metrics["model"]:
        best_pipeline = tuned_pipeline
        best_metrics = {**tuned_metrics, **tuning_summary}
    else:
        best_metrics["tuning_summary"] = tuning_summary

    best_predictions = best_pipeline.predict(X_test)
    report = classification_report(y_test, best_predictions, zero_division=0)
    best_metrics["test_classification_report_path"] = str(CLASSIFICATION_REPORT_PATH)
    best_metrics["optuna_trials_path"] = str(TUNING_TRIALS_PATH)
    best_metrics["optuna_summary_path"] = str(TUNING_SUMMARY_PATH)

    return leaderboard, best_metrics, best_pipeline, report, trials_df


def save_training_outputs(
    leaderboard: pd.DataFrame,
    best_metrics: dict[str, float | str],
    best_pipeline: Pipeline,
    report: str,
    trials_df: pd.DataFrame,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    leaderboard.to_csv(LEADERBOARD_PATH, index=False)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    BEST_MODEL_INFO_PATH.write_text(json.dumps(best_metrics, indent=2))
    CLASSIFICATION_REPORT_PATH.write_text(report)
    TUNING_SUMMARY_PATH.write_text(json.dumps(best_metrics.get("tuning_summary", best_metrics), indent=2))
    trials_df.to_csv(TUNING_TRIALS_PATH, index=False)


def load_best_model(path: Path = BEST_MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"Best model not found at {path}. Run python src/train_model.py first."
        )
    return joblib.load(path)
