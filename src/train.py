import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_loader import load_data
from src.preprocess import clean_text
from src.feature_engineering import create_vectorizer
from src.model_selection import get_models

from config.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_DIR
)


def train_models():
    df = load_data()

    df["message"] = df["message"].apply(clean_text)

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    vectorizer = create_vectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = get_models()

    results = []

    best_model = None
    best_accuracy = 0

    for name, model in models.items():

        model.fit(X_train_vec, y_train)

        predictions = model.predict(X_test_vec)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy
            }
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    os.makedirs(MODEL_DIR, exist_ok=True)

    pd.DataFrame(results).to_csv(
        "outputs/reports/model_metrics.csv",
        index=False
    )

    joblib.dump(
        vectorizer,
        f"{MODEL_DIR}/tfidf_vectorizer.pkl"
    )

    joblib.dump(
        best_model,
        f"{MODEL_DIR}/best_model.pkl"
    )

    return best_model, vectorizer