from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _build_text(df: pd.DataFrame) -> list[str]:
    return (
        df["title"].fillna("").astype(str)
        + "\n"
        + df["text"].fillna("").astype(str)
        + "\nsource:"
        + df["source"].fillna("").astype(str)
        + "\ndomain:"
        + df["domain"].fillna("").astype(str)
        + "\ncommunity:"
        + df["community"].fillna("").astype(str)
        + "\nlang:"
        + df["lang"].fillna("").astype(str)
    ).tolist()


@dataclass
class EmbeddingIndex:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    item_vecs: np.ndarray  # (n_items, d) normalized
    nn: NearestNeighbors

    def query(self, text: str, top_k: int = 50) -> tuple[np.ndarray, np.ndarray]:
        x = self.vectorizer.transform([text])
        z = self.svd.transform(x)
        z = normalize(z, norm="l2")
        dists, idxs = self.nn.kneighbors(z, n_neighbors=min(top_k, self.item_vecs.shape[0]))
        return idxs[0], dists[0]


def build_index(df: pd.DataFrame, n_components: int = 128) -> EmbeddingIndex:
    corpus = _build_text(df)
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    x = vectorizer.fit_transform(corpus)

    n_components = int(min(n_components, max(2, x.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    z = svd.fit_transform(x)
    z = normalize(z, norm="l2")

    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(z)

    return EmbeddingIndex(vectorizer=vectorizer, svd=svd, item_vecs=z, nn=nn)


def embed_text(index: EmbeddingIndex, text: str) -> np.ndarray:
    x = index.vectorizer.transform([text])
    z = index.svd.transform(x)
    z = normalize(z, norm="l2")
    return z[0]

