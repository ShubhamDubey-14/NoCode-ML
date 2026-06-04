"""Baseline model comparison, Optuna tuning, and pipeline persistence."""

from __future__ import annotations

import uuid
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.core.config import MODEL_DIR, OPTUNA_N_TRIALS, RANDOM_STATE, TEST_SIZE
from app.core.preprocessing import build_preprocessor, encode_target

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Stratified split/CV need >=2 rows per class; tiny classes are dropped first.
MIN_SAMPLES_PER_CLASS = 2


def _relabel_consecutive(y: pd.Series) -> pd.Series:
    """Re-map classes to 0..K-1 after row drops (fixes XGBoost label gaps)."""
    le = LabelEncoder()
    encoded = le.fit_transform(y.astype(str))
    return pd.Series(encoded, index=y.index, name=y.name)


def _prune_rare_classes(
    X: pd.DataFrame, y: pd.Series, min_samples: int = MIN_SAMPLES_PER_CLASS
) -> tuple[pd.DataFrame, pd.Series, list[Any]]:
    """Remove rows whose target class has fewer than min_samples examples."""
    counts = y.value_counts()
    rare_labels = counts[counts < min_samples].index.tolist()
    if not rare_labels:
        return X, y, []

    keep = ~y.isin(rare_labels)
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    return X, y, rare_labels


def _use_stratified_split(y: pd.Series) -> bool:
    """Stratify only when every remaining class has enough rows for train/test."""
    if y.nunique() < 2:
        return False
    counts = y.value_counts()
    # With 20% test, classes with only 2 rows can still break stratify; require 5+.
    return int(counts.min()) >= 5


def _cv_folds(y: pd.Series, default: int = 3) -> int:
    """Pick CV folds that no class cannot support."""
    min_count = int(y.value_counts().min())
    return max(2, min(default, min_count))


def _baseline_scores(X_train, X_test, y_train, y_test) -> dict[str, float]:
    """Quick validation accuracy for three classifiers."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
    }
    scores: dict[str, float] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores[name] = float(accuracy_score(y_test, preds))
    return scores


def _pick_winner(baselines: dict[str, float]) -> str:
    return max(baselines, key=baselines.get)  # type: ignore[arg-type]


def _tune_and_fit(
    winner: str,
    X_train,
    y_train,
) -> tuple[Any, dict[str, Any]]:
    """Run Optuna (5 trials) and return fitted estimator + best params."""

    def objective(trial: optuna.Trial) -> float:
        if winner == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model = RandomForestClassifier(
                **params, random_state=RANDOM_STATE, n_jobs=-1
            )
        elif winner == "xgboost":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            }
            model = XGBClassifier(
                n_estimators=100,
                **params,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            C = trial.suggest_float("C", 0.01, 10.0, log=True)
            model = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE)

        from sklearn.model_selection import cross_val_score

        n_splits = _cv_folds(pd.Series(y_train))
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=n_splits, scoring="accuracy"
        )
        return float(cv_scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

    best = study.best_params
    if winner == "random_forest":
        estimator = RandomForestClassifier(
            **best, random_state=RANDOM_STATE, n_jobs=-1
        )
    elif winner == "xgboost":
        estimator = XGBClassifier(
            n_estimators=100,
            **best,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        )
    else:
        estimator = LogisticRegression(
            C=best.get("C", 1.0), max_iter=1000, random_state=RANDOM_STATE
        )

    estimator.fit(X_train, y_train)
    return estimator, best


def train_automl_pipeline(
    df: pd.DataFrame,
    target_column: str,
    imputation_strategy: str,
) -> dict[str, Any]:
    """
    Full AutoML flow: preprocess → baselines → Optuna → save Pipeline.
    Returns metrics and model_id for the API layer.
    """
    X, y_raw, preprocessor = build_preprocessor(df, target_column, imputation_strategy)
    y, _target_encoder = encode_target(y_raw)

    if y.nunique() < 2:
        raise ValueError("Target must have at least two classes for classification.")

    X, y, dropped_classes = _prune_rare_classes(X, y)
    warnings: list[str] = []
    if dropped_classes:
        shown = [str(c) for c in dropped_classes[:15]]
        extra = len(dropped_classes) - len(shown)
        suffix = f" (+{extra} more)" if extra > 0 else ""
        warnings.append(
            f"Removed {len(dropped_classes)} rare class(es) with fewer than "
            f"{MIN_SAMPLES_PER_CLASS} rows each: {', '.join(shown)}{suffix}."
        )

    if y.nunique() < 2:
        raise ValueError(
            "After removing rare classes, fewer than 2 classes remain. "
            "Pick a target with more repeated labels per class, or use a simpler target."
        )

    if len(y) < 10:
        raise ValueError(
            f"Need at least 10 rows after cleaning; only {len(y)} remain. "
            "Use a larger dataset or a different target column."
        )

    y = _relabel_consecutive(y)

    stratify = y if _use_stratified_split(y) else None
    if stratify is None and y.nunique() >= 2:
        warnings.append(
            "Used random (non-stratified) train/test split because some classes "
            "have very few rows."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    # Fit preprocessor on train only, transform both splits
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    baselines = _baseline_scores(X_train_t, X_test_t, y_train, y_test)
    winner = _pick_winner(baselines)

    estimator, best_params = _tune_and_fit(winner, X_train_t, y_train)

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    # Refit on full training split with raw features (pipeline handles transform)
    full_pipeline.fit(X_train, y_train)

    y_pred = full_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }

    model_id = str(uuid.uuid4())
    model_path = MODEL_DIR / f"{model_id}.pkl"
    joblib.dump(full_pipeline, model_path)

    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    return {
        "model_id": model_id,
        "winner": display_names.get(winner, winner),
        "warnings": warnings,
        "rows_used": len(y),
        "classes_used": int(y.nunique()),
        "best_params": best_params,
        "baseline_scores": {
            "Logistic Regression": baselines["logistic_regression"],
            "Random Forest": baselines["random_forest"],
            "XGBoost": baselines["xgboost"],
        },
        "metrics": metrics,
    }
