# Kaggle Disaster Tweets NLP

This project is a Kaggle NLP practice project based on the Disaster Tweets dataset.

The task is to predict whether a tweet is about a real disaster. The main goal of this project is to build a reproducible NLP workflow, starting from a traditional TF-IDF baseline and later extending to Transformer-based models such as BERT.

## Current Progress

- Built a TF-IDF + Logistic Regression baseline.
- Compared several feature variants, including keyword usage, stop-word settings, regularization tuning, and threshold tuning.
- Organized TF-IDF experiments into a unified training script.
- Recorded local cross-validation results and Kaggle public scores for experiment tracking.

## Project Structure

```text
.
├── data/
├── scripts/
│   ├── train_tfidf.py
│   └── archive/
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── features.py
│   └── model.py
├── README.md
├── pyproject.toml
└── .gitignore
```

## How to Run

Run a TF-IDF experiment:

```bash
python scripts/train_tfidf.py --experiment v1_baseline
```

Generate a submission file:

```bash
python scripts/train_tfidf.py --experiment v1_baseline --make-submission
```

## Notes

The current best public-score model is the text-only TF-IDF + Logistic Regression baseline. Further work will focus on building a Transformer-based baseline and comparing it with the traditional NLP baseline.
