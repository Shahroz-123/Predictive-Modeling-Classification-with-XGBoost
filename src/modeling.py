"""Data loading, model pipelines, and evaluation for the project."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


RANDOM_STATE = 42


def _preprocessing_steps() -> list[tuple[str, object]]:
    """Create fresh preprocessing steps for each independent model."""
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]


def load_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load breast-cancer features and labels bundled with scikit-learn."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    # The built-in dataset labels malignant as 0 and benign as 1. Make the
    # positive class (1) mean malignant so recall has a clear interpretation.
    target = (dataset.target == 0).astype(int).rename("malignant")
    return features, target


def load_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the diabetes progression dataset bundled with scikit-learn."""
    dataset = load_diabetes(as_frame=True)
    return dataset.data, dataset.target.rename("progression")


def _classification_pipeline(model: object) -> Pipeline:
    return Pipeline([*_preprocessing_steps(), ("model", model)])


def _regression_pipeline(model: object) -> Pipeline:
    return Pipeline([*_preprocessing_steps(), ("model", model)])


def evaluate_classification() -> pd.DataFrame:
    """Compare XGBoost with a majority-class baseline on held-out data."""
    features, target = load_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    models = {
        "Dummy baseline": _classification_pipeline(
            DummyClassifier(strategy="most_frequent")
        ),
        "XGBoost": _classification_pipeline(
            XGBClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=2,
            )
        ),
    }

    results = []
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results.append(
            {
                "task": "classification",
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
    return pd.DataFrame(results)


def evaluate_regression() -> pd.DataFrame:
    """Compare XGBoost with a mean-prediction baseline on held-out data."""
    features, target = load_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Dummy baseline": _regression_pipeline(DummyRegressor(strategy="mean")),
        "XGBoost": _regression_pipeline(
            XGBRegressor(
                n_estimators=200,
                max_depth=2,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=2,
            )
        ),
    }

    results = []
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results.append(
            {
                "task": "regression",
                "model": name,
                "rmse": mean_squared_error(y_test, predictions) ** 0.5,
                "r2": r2_score(y_test, predictions),
            }
        )
    return pd.DataFrame(results)


def run_and_save_results(output_path: Path) -> pd.DataFrame:
    """Run both tasks, save their test metrics, and return the combined table."""
    results = pd.concat(
        [evaluate_classification(), evaluate_regression()],
        ignore_index=True,
        sort=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results
