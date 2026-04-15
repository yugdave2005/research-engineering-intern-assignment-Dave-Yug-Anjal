"""
clustering.py — BERTopic-based topic clustering with tunable n_topics.
Embedding visualization via UMAP + datamapplot.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_PATH = Path(__file__).parent.parent / "cache"


def cluster_topics(df: pd.DataFrame, n_topics: int = 10, text_col: str = "full_text", embeddings: np.ndarray = None) -> pd.DataFrame:
    """
    Cluster posts by topic using BERTopic with UMAP + HDBSCAN.
    n_topics: target number of topics (2–50).
    embeddings: Optional pre-computed embeddings for the sampled texts.
    Returns result_df, model.
    """
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    texts = df[text_col].fillna("").astype(str).tolist()

    # Clamp n_topics to valid range
    n_topics = max(2, min(n_topics, min(50, len(texts) // 5)))

    # UMAP params (dimensionality reduction for clustering)
    umap_model = UMAP(
        n_neighbors=min(15, len(texts) - 1),
        n_components=5,  # Higher components for clustering, px.scatter will re-run UMAP to 2D
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN (density-based clustering)
    # min_cluster_size is the most important parameter for target topic count
    target_cluster_size = max(10, len(texts) // (n_topics * 1.5))
    hdbscan_model = HDBSCAN(
        min_cluster_size=int(target_cluster_size),
        min_samples=2,
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

    topics, _ = model.fit_transform(texts, embeddings=embeddings)

    # Get 2D UMAP embeddings for visualization (re-run on the embeddings for better 2D positioning)
    try:
        if embeddings is not None:
            # Fastest: use the pre-computed embeddings we were passed
            vis_umap = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=42)
            embs_2d = vis_umap.fit_transform(embeddings)
        else:
            # Fallback: let BERTopic's internal UMAP (which we set to 5D above) handle it or re-run
            vis_umap = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=42)
            # We need the transformed embeddings from the model's perspective
            # But simple fit_transform on texts would re-encode. 
            # If we reached here, embeddings was None, so Fit transform already encoded.
            # BERTopic stores embeddings in model._embeddings after fit_transform.
            if hasattr(model, "_embeddings") and model._embeddings is not None:
                embs_2d = vis_umap.fit_transform(model._embeddings)
            else:
                embs_2d = np.zeros((len(texts), 2))
    except Exception as e:
        print(f"Vis UMAP failed: {e}")
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
