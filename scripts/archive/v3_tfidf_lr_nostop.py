from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.model import build_tfidf_logistic_regression_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["text"]
    y_train = train_df["target"]
    X_test = test_df["text"]

    model = build_tfidf_logistic_regression_model(stop_words=None)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    scores = cross_val_score(model,
                             X_train,
                             y_train,
                             cv=cv,
                             scoring="f1",
                             n_jobs=-1)
    print(scores)
    print("mean_f1:", scores.mean())
    print("std_f1:", scores.std())

    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "target": test_preds,
    })

    submission_path = OUTPUT_DIR / "submission_v3_tfidf_lr_nostop.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved submission to: {submission_path}")


if __name__ == "__main__":
    main()
