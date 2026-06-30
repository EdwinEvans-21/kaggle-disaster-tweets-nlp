from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.model import build_tfidf_logistic_regression_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def find_best_threshold(model, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    oof_probs = np.zeros(len(y))

    for train_idx, val_idx in cv.split(X, y):
        model_fold = clone(model)

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        model_fold.fit(X_train, y_train)

        val_probs = model_fold.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs

    thresholds = np.arange(0.30, 0.71, 0.01)

    results = []

    for threshold in thresholds:
        oof_preds = (oof_probs >= threshold).astype(int)
        score = f1_score(y, oof_preds)

        results.append({
            "threshold": float(threshold),
            "f1": float(score),
        })

    results_df = pd.DataFrame(results).sort_values(
        by="f1",
        ascending=False,
    )

    best_threshold = results_df.iloc[0]["threshold"]
    best_f1 = results_df.iloc[0]["f1"]

    return best_threshold, best_f1, results_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["text"].fillna("")
    y_train = train_df["target"]
    X_test = test_df["text"]

    model = build_tfidf_logistic_regression_model()
    best_threshold, best_f1, threshold_results = find_best_threshold(
        model,
        X_train,
        y_train,
    )

    print("best_threshold:", best_threshold)
    print("best_oof_f1:", best_f1)
    print(threshold_results.head(10))

    model = build_tfidf_logistic_regression_model()

    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "target": test_preds,
    })

    threshold_tag = f"thr{int(best_threshold * 100):03d}"
    submission_path = OUTPUT_DIR / f"submission_v7_tfidf_lr_text_{threshold_tag}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved submission to: {submission_path}")


if __name__ == "__main__":
    main()
