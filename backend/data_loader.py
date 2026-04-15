"""
data_loader.py — Load & normalize Reddit data from data.jsonl
Fast: parses JSONL once, caches to Parquet; subsequent loads are near-instant.
"""
import json
import pandas as pd
from pathlib import Path

DATA_PATH  = Path(__file__).parent.parent / "data.jsonl"
CACHE_PATH = Path(__file__).parent.parent / "cache"
PARQUET    = CACHE_PATH / "data.parquet"

# Columns we actually need (skip the 5000+ nested media_metadata columns)
KEEP_COLS = [
    "id", "subreddit", "author", "author_fullname", "title", "selftext",
    "score", "upvote_ratio", "num_comments", "created",
    "domain", "url", "is_self", "link_flair_text",
    "over_18", "locked", "gilded", "total_awards_received",
    "thumbnail", "permalink", "name",
]


def load_data() -> pd.DataFrame:
    """Load data.jsonl → parquet cache on first run, then read parquet (fast)."""
    CACHE_PATH.mkdir(exist_ok=True)

    if PARQUET.exists():
        # Fast path: ~0.1s
        df = pd.read_parquet(PARQUET)
        return df

    # --- First run: parse JSONL (~3-5s, done once) ---
    records = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                d = obj.get("data", obj)
                records.append({k: d.get(k) for k in KEEP_COLS})
            except Exception:
                pass

    df = pd.DataFrame(records)

    # ── Timestamps ──────────────────────────────────────────────────────────
    df["created_dt"] = pd.to_datetime(df["created"], unit="s", utc=True, errors="coerce")
    df["date"] = df["created_dt"].dt.date
    # Strip tz before Period conversion to avoid UserWarning
    dt_naive = df["created_dt"].dt.tz_localize(None)
    df["year_month"] = dt_naive.dt.to_period("M").astype(str)
    df["week"] = dt_naive.dt.to_period("W").astype(str)

    # ── Text content ─────────────────────────────────────────────────────────
    df["title"] = df["title"].fillna("").astype(str)
    df["selftext"] = df["selftext"].fillna("").astype(str)
    df["full_text"] = (df["title"] + " " + df["selftext"]).str.strip()

    # ── Numerics ──────────────────────────────────────────────────────────────
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0).astype(int)
    df["num_comments"] = pd.to_numeric(df.get("num_comments"), errors="coerce").fillna(0).astype(int)
    df["upvote_ratio"] = pd.to_numeric(df.get("upvote_ratio"), errors="coerce").fillna(0.5)

    # ── Domain / URL ─────────────────────────────────────────────────────────
    df["domain"] = df.get("domain", "").fillna("").astype(str)
    df["url"] = df.get("url", "").fillna("").astype(str)
    df["is_external"] = ~df["domain"].str.startswith("self.")

    # ── Author / Subreddit ────────────────────────────────────────────────────
    df["author"] = df.get("author", "[deleted]").fillna("[deleted]").astype(str)
    df["subreddit"] = df.get("subreddit", "unknown").fillna("unknown").astype(str)

    # ── Hashtags / flairs (treat link_flair_text as topic tag) ──────────────
    df["flair"] = df.get("link_flair_text", "").fillna("None").astype(str)

    # ── Drop rows with missing critical fields ───────────────────────────────
    df = df.dropna(subset=["created_dt", "subreddit"])
    df = df.reset_index(drop=True)

    # Save to parquet for fast future loads
    df.to_parquet(PARQUET, index=False)

    return df


# ── Singleton cache ───────────────────────────────────────────────────────────
_df_cache = None


def get_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = load_data()
    return _df_cache
