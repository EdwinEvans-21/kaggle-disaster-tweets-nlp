from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_logistic_regression_model(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=False,
        C=1.0,
):
    model = Pipeline(steps=[
        (
            "tf_idf",
            TfidfVectorizer(lowercase=True,
                            stop_words=stop_words,
                            ngram_range=ngram_range,
                            min_df=min_df,
                            max_df=max_df,
                            sublinear_tf=sublinear_tf),
        ),
        (
            "classifier",
            LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    return model
