"""
clustering.py — BERTopic-based topic clustering with tunable n_topics.
Embedding visualization via UMAP + datamapplot.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_PATH = Path(__file__).parent.parent / "cache"


def cluster_topics(df: pd.DataFrame, n_topics: int = 10, text_col: str = "full_text") -> pd.DataFrame:
    """
    Cluster posts by topic using BERTopic with UMAP + HDBSCAN.
    n_topics: target number of topics (2–50). Handles extremes without crashing.
    Returns df with added columns: topic_id, topic_label, umap_x, umap_y.
    """
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    texts = df[text_col].fillna("").tolist()

    # Clamp n_topics to valid range
    n_topics = max(2, min(n_topics, min(50, len(texts) // 10)))

    # UMAP params
    umap_model = UMAP(
        n_neighbors=min(15, len(texts) - 1),
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(5, len(texts) // (n_topics * 3)),
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=n_topics,
        verbose=False,
        calculate_probabilities=False,
    )

    topics, _ = model.fit_transform(texts)

    # Get 2D UMAP embeddings for visualization
    try:
        embs_2d = model.umap_model.embedding_
    except Exception:
        embs_2d = np.zeros((len(texts), 2))

    # Get topic labels (top 3 words)
    topic_info = model.get_topic_info()
    label_map = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            label_map[tid] = "Miscellaneous"
        else:
            top_words = [w for w, _ in (model.get_topic(tid) or [])][:3]
            label_map[tid] = " · ".join(top_words) if top_words else f"Topic {tid}"

    result = df.copy()
    result["topic_id"]    = topics
    result["topic_label"] = [label_map.get(t, f"Topic {t}") for t in topics]
    result["umap_x"]      = embs_2d[:, 0] if embs_2d.shape[1] >= 1 else 0.0
    result["umap_y"]      = embs_2d[:, 1] if embs_2d.shape[1] >= 2 else 0.0

    return result, model


def get_topic_summary(model, n_topics: int = 10) -> pd.DataFrame:
    """Return a summary DataFrame: topic_id, label, size, top_words."""
    info = model.get_topic_info()
    rows = []
    for _, row in info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            label = "Miscellaneous"
            top_words = []
        else:
            top_words = [w for w, _ in (model.get_topic(tid) or [])][:5]
            label = " · ".join(top_words[:3]) if top_words else f"Topic {tid}"
        rows.append({
            "topic_id": tid,
            "label": label,
            "size": row["Count"],
            "top_words": ", ".join(top_words),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False)
