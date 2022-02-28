import joblib
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

PROJECT_DIR = Path(__file__).parent.parent
BASE_DATA_DIR = PROJECT_DIR / "data"
BASE_MODEL_DIR = PROJECT_DIR / "models"
BASE_FIGURES_DIR = PROJECT_DIR / "figures"


def _make_dataframe(
    articles_df: pd.DataFrame, root_dir: Path = BASE_DATA_DIR
) -> pd.DataFrame:
    """Reads articles to memory and adds them to dataframe.

    Modifies dataframe in place (adds/replaces "text" column).

    Args:
        articles_df (pd.DataFrame): Dataframe with articles to read
        root_dir (Path, optional): Root dir of data. Paths in df
            will be relative to this.
            Defaults to BASE_DATA_DIR.

    Returns:
        pd.DataFrame: Dataframe with article text as "text" column
    """

    def read_file(path: Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    articles_df = articles_df.copy()

    text_list = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(read_file)(root_dir / path)
        for path in tqdm(articles_df.path, total=articles_df.shape[0])
    )
    articles_df["text"] = np.array(text_list, dtype=object)
    return articles_df


def split_dataframe(
    articles_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    source_thresh_reliable: float = 2,
    source_thresh_unreliable: float = -2,
    num_val_sources: int = 2,
) -> pd.DataFrame:
    """Splits dataframe into train, val and pred.

    Modifies dataframe in place (adds/replaces "split" column).

    Args:
        articles_df (pd.DataFrame): Dataframe of articles to be split
        scores_df (pd.DataFrame): Reliablility scores of sources
        source_thresh_reliable (float, optional): Every article from a source
            scoring this high or higher will be considered true news.
            Defaults to 2.5.
        source_thresh_unreliable (float, optional): Every article from a source
            scoring this or lower will be considered fake news. Defaults to -2.
        num_val_sources (int, optional): Number of sources to use for validation.

    Returns:
        pd.DataFrame: original articles dataframe with "split" column added
    """
    # The logic here will need to be revisited
    # I just played with numbers until I got val split of decent size and balance
    reliable_sources = scores_df[scores_df.score >= source_thresh_reliable]
    unreliable_sources = scores_df[scores_df.score <= source_thresh_unreliable]

    val_unreliable_sources = unreliable_sources.iloc[:num_val_sources]
    train_unreliable_sources = unreliable_sources.iloc[num_val_sources:]

    val_reliable_sources = reliable_sources.iloc[-num_val_sources - 1 :]
    train_reliable_sources = reliable_sources.iloc[: -num_val_sources - 1]

    train_sources = pd.concat([train_reliable_sources, train_unreliable_sources])
    val_sources = pd.concat([val_reliable_sources, val_unreliable_sources])

    articles_df["split"] = "pred"
    articles_df.loc[articles_df.source.isin(train_sources.index), "split"] = "train"
    articles_df.loc[articles_df.source.isin(val_sources.index), "split"] = "val"

    num_articles_train_reliable = sum(
        (articles_df.split == "train")
        & (articles_df.source_score >= source_thresh_reliable)
    )
    num_articles_train_unreliable = sum(
        (articles_df.split == "train")
        & (articles_df.source_score < source_thresh_reliable)
    )
    num_articles_val_reliable = sum(
        (articles_df.split == "val")
        & (articles_df.source_score >= source_thresh_reliable)
    )
    num_articles_val_unreliable = sum(
        (articles_df.split == "val")
        & (articles_df.source_score < source_thresh_reliable)
    )

    print(
        "We have:\n"
        f"{len(train_reliable_sources)} reliable sources in train\n"
        f"{len(train_unreliable_sources)} unreliable sources in train\n"
        f"For a total of "
        f"{num_articles_train_reliable}+"
        f"{num_articles_train_unreliable}="
        f"{sum(articles_df.split == 'train')} articles in train\n"
        f"{len(val_reliable_sources)} reliable sources in val\n"
        f"{len(val_unreliable_sources)} unreliable sources in val\n"
        f"For a total of "
        f"{num_articles_val_reliable}+"
        f"{num_articles_val_unreliable}="
        f"{sum(articles_df.split == 'val')} articles in val\n"
    )

    return articles_df


def _balance_by_dropping(dataframe):
    """Drops articles from dataframe to balance the dataset."""
    real = dataframe[~dataframe.is_fake]
    fake = dataframe[dataframe.is_fake]
    # I have way more articles of real news
    drop_real = max(0, real.shape[0] - fake.shape[0])
    real = real.sample(n=real.shape[0] - drop_real, random_state=42)
    print(f"Dropped {drop_real} real articles")
    return pd.concat([fake, real]).sort_index()


def make_train_dataframes(
    articles_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    source_thresh_reliable: float = 2,
    source_thresh_unreliable: float = -2,
    num_val_sources: int = 2,
    balance: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataframe into train, val and pred. Reads articles to memory.

    Original dataframe is modified in place: "split" column is added.

    Args:
        Most of the arguments are passed to split_dataframe. See its docstring.

        balance (bool, optional): Whether to balance the dataset by dropping
            exess articles.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): train and val dataframes
    """
    articles_df["is_fake"] = articles_df.source_score < 0
    articles_df = split_dataframe(
        articles_df,
        scores_df,
        source_thresh_reliable,
        source_thresh_unreliable,
        num_val_sources,
    )

    print("Processing train articles")
    train_split = articles_df[articles_df.split == "train"]
    if balance:
        train_split = _balance_by_dropping(train_split)
    print("Reading train articles")
    train_articles_df = _make_dataframe(train_split)
    train_articles_df = train_articles_df.sample(frac=1, random_state=42)  # shuffle

    print("Processing val articles")
    val_split = articles_df[articles_df.split == "val"]
    if balance:
        val_split = _balance_by_dropping(val_split)
    print("Reading val articles")
    val_articles_df = _make_dataframe(val_split)

    return train_articles_df, val_articles_df
