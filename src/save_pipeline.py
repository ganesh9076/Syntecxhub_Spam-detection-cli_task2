import joblib

from sklearn.pipeline import Pipeline

from src.feature_engineering import create_vectorizer


def save_pipeline():
    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    model = joblib.load(
        "models/best_model.pkl"
    )

    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("model", model)
        ]
    )

    joblib.dump(
        pipeline,
        "models/complete_pipeline.pkl"
    )