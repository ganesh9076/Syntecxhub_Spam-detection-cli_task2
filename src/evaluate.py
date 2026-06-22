import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocess import clean_text

from config.config import TEST_SIZE, RANDOM_STATE


def evaluate_model():
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

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    model = joblib.load(
        "models/best_model.pkl"
    )

    X_test_vec = vectorizer.transform(X_test)

    predictions = model.predict(X_test_vec)

    report = classification_report(
        y_test,
        predictions
    )

    with open(
        "outputs/reports/classification_report.txt",
        "w"
    ) as f:
        f.write(report)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            pos_label="spam"
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            pos_label="spam"
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            pos_label="spam"
        )
    }

    cm = confusion_matrix(y_test, predictions)

    return metrics, cm