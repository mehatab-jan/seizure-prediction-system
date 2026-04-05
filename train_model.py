from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train seizure risk model from tabular features.")
    parser.add_argument("--data", required=True, help="Path to CSV file.")
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=["label", "target", "seizure", "risk"],
        help="Target column candidates in priority order.",
    )
    parser.add_argument("--output", default="model_bundle.pkl", help="Output model file.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def find_target_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of target columns found. Tried: {candidates}")


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    target_col = find_target_column(df, args.target_columns)

    y = df[target_col].astype(int).to_numpy()
    X = df.drop(columns=[target_col])

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_features:
        raise ValueError("No numeric features found in input CSV.")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    base_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", CalibratedClassifierCV(base_model, method="isotonic", cv=3)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    threshold, best_f1 = find_best_threshold(y_test, proba)
    preds = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(best_f1),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }

    bundle = {
        "model": model,
        "feature_names": numeric_features,
        "threshold": threshold,
        "metrics": metrics,
        "metadata": {
            "dataset": str(data_path),
            "rows": int(len(df)),
            "positive_rate": float(np.mean(y)),
        },
    }

    joblib.dump(bundle, args.output)

    print("Saved model bundle to", args.output)
    print("Target column:", target_col)
    print("Threshold:", round(threshold, 3))
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key:10s}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
