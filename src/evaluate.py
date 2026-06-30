import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_cv(model, X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )

    return scores


def find_best_threshold(model, X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_probs = np.zeros(len(y))

    for train_idx, val_idx in cv.split(X, y):
        model_fold = clone(model)

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        model_fold.fit(X_train, y_train)
        oof_probs[val_idx] = model_fold.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.30, 0.71, 0.01)

    results = []

    for threshold in thresholds:
        preds = (oof_probs >= threshold).astype(int)
        score = f1_score(y, preds)

        results.append(
            {
                "threshold": float(threshold),
                "f1": float(score),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by="f1",
        ascending=False,
    )

    best_threshold = results_df.iloc[0]["threshold"]
    best_f1 = results_df.iloc[0]["f1"]

    return best_threshold, best_f1, results_df