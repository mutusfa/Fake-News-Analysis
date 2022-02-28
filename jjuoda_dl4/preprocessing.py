"""Collection of functions for preprocessing data."""

import fasttext.FastText
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset

from .utils import BASE_DATA_DIR

# I could fix this to use joblib, but I couldn't do it quickly and w/e, let's get this
# project finished first.
def encode_articles(
    articles_df: pd.DataFrame, ft_encoder: fasttext.FastText._FastText
) -> np.ndarray:
    """Encodes articles using fasttext.

    Encodings for for title are calculated separately and then
    concatenated, in effect saying that article's title is as important
    as its whole body.

    Args:
        articles_df (pd.DataFrame): Dataframe of articles to encode
            Needs to have "title" and "text" columns.
        ft_encoder (fasttext.FastText._FastText): Fasttext encoder

    Returns:
        600-dimensional numpy array of encoded articles;
            0:300 features represent title
            300: features represent text of the article
    """
    encodings_text = np.stack(
        [
            ft_encoder.get_sentence_vector(s.lower().replace("\n", " "))
            for s in articles_df.text
        ]
    )
    encodings_title = np.stack(
        [
            ft_encoder.get_sentence_vector(s.lower().replace("\n", " "))
            for s in articles_df.title
        ]
    )
    return np.concatenate([encodings_text, encodings_title], axis=1)


def get_tfidf_vectors(
    articles_df: pd.DataFrame,
    title_tfidf_vectorizer: TfidfVectorizer,
    text_tfidf_vectorizer: TfidfVectorizer = None,
):
    """Gets tfidf vectors for articles."""
    if text_tfidf_vectorizer is None:
        text_tfidf_vectorizer = title_tfidf_vectorizer
    title_vectors = title_tfidf_vectorizer.transform(articles_df.title).todense()
    text_vectors = text_tfidf_vectorizer.transform(articles_df.text).todense()
    return np.array(np.concatenate([title_vectors, text_vectors], axis=1))


class AutoNLPNELADataset(Dataset):
    """Recreates processeing I did for AutoNLP on nela gt 2018 data."""

    def __init__(self, articles_df, tokenizer, root_dir=BASE_DATA_DIR):
        self.articles_df = articles_df
        self.root_dir = root_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.articles_df)

    def __getitem__(self, index):
        article = self.articles_df.iloc[index]

        if "text" in article.keys():
            text = article["text"].replace("\n", " ")
        else:
            with open(self.root_dir / article.path, "r") as f:
                text = f.read().replace("\n", " ")
        text = "<TITLE>" + article.title + "</TITLE> " + text

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation="longest_first",
        )

        # source score ain't needed for prediction, it just helped to make graphs easier
        if "source_score" in article.keys():
            source_score = article["source_score"]
        else:
            source_score = np.nan

        return {
            "model_inputs": inputs,
            "source_score": source_score,
            # this one is actually needed for training
            "is_fake": source_score < 0,
        }
