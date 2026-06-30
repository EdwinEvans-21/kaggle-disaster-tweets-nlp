import argparse
from pathlib import Path

import pandas as pd

from src.evaluate import evaluate_cv, find_best_threshold
from src.features import build_input_text
from src.model import build_tfidf_logistic_regression_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

EXPERIMENTS = {
    "v1_baseline": {
        "use_keyword": False,
        "stop_words": "english",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": False,
        "C": 1.0,
        "threshold": 0.5,
    },
    "v2_kw": {
        "use_keyword": True,
        "stop_words": "english",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": False,
        "C": 1.0,
        "threshold": 0.5,
    },
    "v3_nostop": {
        "use_keyword": False,
        "stop_words": None,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": False,
        "C": 1.0,
        "threshold": 0.5,
    },
    "v4_kw_nostop": {
        "use_keyword": True,
        "stop_words": None,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": False,
        "C": 1.0,
        "threshold": 0.5,
    },
    "v6_kw_nostop_c1p5": {
        "use_keyword": True,
        "stop_words": None,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": False,
        "C": 1.5,
        "threshold": 0.5,
    },
    "v7_text_thr045": {
        "use_keyword": False,
        "stop_words": "english",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": False,
        "C": 1.0,
        "threshold": 0.45,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=EXPERIMENTS.keys(),
    )

    parser.add_argument(
        "--make-submission",
        action="store_true",
    )

    parser.add_argument(
        "--tune-threshold",
        action="store_true",
    )

    return parser.parse_args()


def build_model(config):
    return build_tfidf_logistic_regression_model(
        stop_words=config["stop_words"],
        ngram_range=config["ngram_range"],
        min_df=config["min_df"],
        max_df=config["max_df"],
        sublinear_tf=config["sublinear_tf"],
        C=config["C"],
    )


def make_submission(model, X, y, X_test, test_df, experiment_name, threshold):
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    model.fit(X, y)

    if threshold == 0.5:
        test_preds = model.predict(X_test)
        threshold_tag = "default"
    else:
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= threshold).astype(int)
        threshold_tag = f"thr{int(threshold * 100):03d}"

    submission = pd.DataFrame({
        "id": test_df["id"],
        "target": test_preds,
    })

    submission_path = SUBMISSION_DIR / f"submission_{experiment_name}_{threshold_tag}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved submission to: {submission_path}")


def main():
    args = parse_args()
    config = EXPERIMENTS[args.experiment]

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    X = build_input_text(train_df, use_keyword=config["use_keyword"])
    y = train_df["target"]
    X_test = build_input_text(test_df, use_keyword=config["use_keyword"])

    model = build_model(config)

    scores = evaluate_cv(model, X, y)

    print(f"Experiment: {args.experiment}")
    print(f"CV scores: {scores}")
    print(f"mean_f1: {scores.mean():.5f}")
    print(f"std_f1: {scores.std():.5f}")

    threshold = config["threshold"]

    if args.tune_threshold:
        best_threshold, best_f1, threshold_results = find_best_threshold(
            model, X, y)
        print(f"best_threshold: {best_threshold:.2f}")
        print(f"best_oof_f1: {best_f1:.5f}")
        print(threshold_results.head(10))

        threshold = best_threshold

    if args.make_submission:
        make_submission(
            model=model,
            X=X,
            y=y,
            X_test=X_test,
            test_df=test_df,
            experiment_name=args.experiment,
            threshold=threshold,
        )


if __name__ == "__main__":
    main()
