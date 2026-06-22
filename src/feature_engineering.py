from sklearn.feature_extraction.text import TfidfVectorizer
from config.config import MAX_FEATURES, NGRAM_RANGE


def create_vectorizer():
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE
    )