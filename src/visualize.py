import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from src.data_loader import load_data
from src.preprocess import clean_text

from src.evaluate import evaluate_model


def plot_confusion_matrix():
    _, cm = evaluate_model()

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        "outputs/plots/confusion_matrix.png"
    )

    plt.close()


def plot_model_comparison():
    df = pd.read_csv(
        "outputs/reports/model_metrics.csv"
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="Model",
        y="Accuracy"
    )

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/model_comparison.png"
    )

    plt.close()

def generate_wordclouds():

    df = load_data()

    df["message"] = df["message"].apply(clean_text)

    spam_text = " ".join(
        df[df["label"] == "spam"]["message"]
    )

    ham_text = " ".join(
        df[df["label"] == "ham"]["message"]
    )

    spam_cloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(spam_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(spam_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        "outputs/plots/wordcloud_spam.png"
    )

    plt.close()

    ham_cloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(ham_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(ham_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        "outputs/plots/wordcloud_ham.png"
    )
    plt.close()
    