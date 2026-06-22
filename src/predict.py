import csv
import os
import joblib

from datetime import datetime


def predict_message():
    pipeline = joblib.load(
        "models/complete_pipeline.pkl"
    )

    message = input(
        "\nEnter Message:\n"
    )

    prediction = pipeline.predict(
        [message]
    )[0]

    confidence = "N/A"

    try:
        probability = pipeline.named_steps[
            "model"
        ].predict_proba(
            pipeline.named_steps[
                "tfidf"
            ].transform([message])
        )

        confidence = round(
            max(probability[0]) * 100,
            2
        )

    except:
        pass

    print(
        f"\nPrediction : {prediction.upper()}"
    )

    print(
        f"Confidence : {confidence}"
    )

    os.makedirs(
        "outputs/logs",
        exist_ok=True
    )

    file_path = (
        "outputs/logs/prediction_history.csv"
    )

    exists = os.path.isfile(file_path)

    with open(
        file_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not exists:
            writer.writerow(
                [
                    "Timestamp",
                    "Message",
                    "Prediction",
                    "Confidence"
                ]
            )

        writer.writerow(
            [
                datetime.now(),
                message,
                prediction,
                confidence
            ]
        )