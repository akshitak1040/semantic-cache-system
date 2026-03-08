from sklearn.datasets import fetch_20newsgroups
import re


def clean_text(text):
    """
    Basic cleaning for newsgroup posts
    """

    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)

    return text.strip()


def load_documents():
    """
    Load and clean the 20 newsgroups dataset
    """

    dataset = fetch_20newsgroups(subset="all")

    docs = []

    for text in dataset.data:

        cleaned = clean_text(text)

        if len(cleaned) > 100:
            docs.append(cleaned)

    print("Total cleaned documents:", len(docs))

    return docs