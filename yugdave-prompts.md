# yugdave-prompts.md

This file documents all AI-assisted development for this assignment, as required by the SimPPL AI Usage Policy.
Each prompt is numbered sequentially with the component it was for, the prompt itself, and a note on what was
wrong with the output and how it was fixed.

---

## Prompt 1 — Data Loader
**Component:** `backend/data_loader.py`
**Prompt:** "Write a Python function to load a Reddit JSONL file where each line is `{kind, data}` and normalize
it into a flat pandas DataFrame with timestamps, author, subreddit, score, and text columns."
**Fix:** Initial output used `pd.json_normalize` which created 5000+ nested columns. Fixed by manually extracting
only the relevant fields from `obj['data']` and constructing the DataFrame directly.

---

## Prompt 2 — FAISS Semantic Search
**Component:** `backend/embeddings.py`
**Prompt:** "Implement FAISS-backed semantic search using sentence-transformers all-MiniLM-L6-v2. Cache the index
to disk. Handle empty queries, very short queries, and non-English input gracefully."
**Fix:** First version didn't normalize embeddings before adding to FAISS, so cosine similarity scores were wrong.
Fixed by using `normalize_embeddings=True` in `model.encode()` and `IndexFlatIP` (inner product on unit vectors = cosine).

---

## Prompt 3 — Network Graph
**Component:** `backend/network.py`
**Prompt:** "Build a bipartite author-subreddit graph with NetworkX. Compute PageRank, betweenness centrality,
and Louvain community detection. Handle disconnected components without crashing."
**Fix:** `betweenness_centrality` timed out on large graphs. Fixed by using the `k=100` approximation for
graphs with more than 500 nodes.

---

## Prompt 4 — BERTopic Clustering
**Component:** `backend/clustering.py`
**Prompt:** "Use BERTopic with HDBSCAN and UMAP to cluster Reddit posts. Expose n_topics as a tunable parameter.
Handle extremes (n=2, n=50) without crashing."
**Fix:** HDBSCAN `min_cluster_size` was too large and produced only 1 cluster. Fixed by setting
`min_cluster_size = max(5, len(texts) // (n_topics * 3))` to scale with dataset size and target count.

---

## Prompt 5 — Streamlit UI
**Component:** `app.py`
**Prompt:** "Build a dark-themed Streamlit dashboard with 4 tabs: Overview/Trends, Network Analysis,
Semantic Search, and Topic Clusters. Each time-series chart should have a GenAI-generated plain-language
summary below it. The chatbot should suggest 2-3 follow-up queries."
**Fix:** Streamlit's `st.markdown` with `unsafe_allow_html=True` stripped some CSS classes. Fixed by
inlining all critical styles directly in the HTML strings for cards and post displays.

---

## Prompt 6 — LLM Integration
**Component:** `backend/llm.py`
**Prompt:** "Write a Groq API wrapper that generates plain-language chart summaries and chatbot responses.
Fall back to backup API key if the primary fails. Handle API rate limits gracefully."
**Fix:** Initial version raised unhandled exceptions on Groq rate limit errors. Fixed by wrapping each API
call in a try/except loop that tries both keys before falling back to a deterministic string response.

---
