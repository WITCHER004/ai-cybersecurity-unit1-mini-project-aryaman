"""Helpers shared by the task scripts: data loading, splitting, pipelines, metrics."""
import time

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    FIG_DIR,
    NUMERIC_FEATURES,
    SEED,
    TARGET,
    TEST_SIZE,
    ensure_dirs,
)

sns.set_theme(style="whitegrid")


def load_data() -> pd.DataFrame:
    """Load the dataset, generating it first if it does not exist yet."""
    ensure_dirs()
    if not DATA_PATH.exists():
        import generate_dataset

        generate_dataset.main()
    df = pd.read_csv(DATA_PATH)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object)
    return df


def split_data(df: pd.DataFrame):
    """Stratified 75/25 split -> X_train, X_test, y_train, y_test."""
    return train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=SEED,
    )


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def logistic_regression() -> Pipeline:
    return make_pipeline(LogisticRegression(max_iter=1000))


def decision_tree() -> Pipeline:
    return make_pipeline(DecisionTreeClassifier(max_depth=6, random_state=SEED))


def neural_network() -> Pipeline:
    return make_pipeline(
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            early_stopping=True,
            max_iter=300,
            random_state=SEED,
        )
    )


def fit_timed(pipe: Pipeline, X, y) -> float:
    start = time.perf_counter()
    pipe.fit(X, y)
    return time.perf_counter() - start


def evaluate(y_true, y_pred) -> dict:
    """Metrics on the suspicious class (label 1) plus confusion-matrix counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def save_fig(fig, name: str) -> None:
    ensure_dirs()
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.relative_to(FIG_DIR.parent)}")


def fmt_table(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda v: f"{v:.3f}")


__all__ = [
    "np",
    "pd",
    "plt",
    "sns",
    "load_data",
    "split_data",
    "build_preprocessor",
    "make_pipeline",
    "logistic_regression",
    "decision_tree",
    "neural_network",
    "fit_timed",
    "evaluate",
    "save_fig",
    "fmt_table",
]
