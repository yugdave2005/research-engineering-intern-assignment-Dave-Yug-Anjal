"""
embeddings.py — Sentence-transformer embeddings + FAISS semantic search.
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

CACHE_PATH = Path(__file__).parent.parent / "cache"
EMB_FILE   = CACHE_PATH / "embeddings.npy"
IDX_FILE   = CACHE_PATH / "faiss.index"

_model  = None
_index  = None
_ids    = None   # row indices corresponding to embeddings


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_index(df: pd.DataFrame, text_col: str = "full_text") -> None:
    """Encode all posts and build a FAISS index. Cached to disk."""
    import faiss

    CACHE_PATH.mkdir(exist_ok=True)
    model = _get_model()

    texts = df[text_col].fillna("").tolist()
    print(f"[embeddings] Encoding {len(texts)} documents…")
    embs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product == cosine on normalized vectors
    index.add(embs.astype(np.float32))

    np.save(str(EMB_FILE), embs)
    faiss.write_index(index, str(IDX_FILE))
    print("[embeddings] Index built and cached.")


def _load_index():
    global _index
    import faiss
    if _index is None and IDX_FILE.exists():
        _index = faiss.read_index(str(IDX_FILE))
    return _index


def semantic_search(query: str, df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Return the top_k most semantically relevant rows for a query.
    Handles empty queries, very short queries, and non-English input gracefully.
    """
    # Edge case: empty / whitespace-only query
    if not query or not query.strip():
        return pd.DataFrame()

    model  = _get_model()
    index  = _load_index()

    if index is None:
        build_index(df)
        index = _load_index()

    q_emb = model.encode([query.strip()], normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)

    actual_k = min(top_k, index.ntotal)
    if actual_k == 0:
        return pd.DataFrame()

    scores, idxs = index.search(q_emb, actual_k)
    result_df = df.iloc[idxs[0]].copy()
    result_df["relevance_score"] = scores[0]
    return result_df.reset_index(drop=True)
