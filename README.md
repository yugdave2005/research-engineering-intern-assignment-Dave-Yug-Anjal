# 🔬 NarrativeScope — Reddit Political Discourse Analysis

> **SimPPL Research Engineering Intern Assignment**  
> An investigative analysis dashboard for understanding how political narratives spread across Reddit communities (Jul 2024 – Feb 2025).

---

## 🌐 Live Dashboard
<!-- Add the deployed URL here after hosting -->
**Hosted at:** _Coming soon (Streamlit Cloud)_

## 🎥 Demo Video
<!-- Add YouTube/Drive link here -->
**Video walkthrough:** _Coming soon_

---

## 📊 What This Does

NarrativeScope analyzes **8,799 Reddit posts** across **10 political subreddits** (r/neoliberal, r/politics, r/Conservative, r/socialism, r/Anarchism, r/democrats, r/Republican, r/Liberal, r/worldpolitics, r/PoliticalDiscussion).

### Features

| Feature | Description |
|---|---|
| **📈 Time-Series Trends** | Daily/weekly/monthly post volume & score distribution, with AI-generated plain-language summaries |
| **🕸 Network Analysis** | Author–subreddit bipartite graph with PageRank, betweenness centrality, Louvain community detection |
| **🔍 Semantic Search** | Embedding-based search (no keyword matching required). Works with empty/short/non-English queries |
| **🤖 AI Chatbot** | LLM-powered chatbot with follow-up query suggestions, grounded on retrieved posts |
| **🧩 Topic Clustering** | BERTopic with tunable cluster count (2–50). UMAP 2D embedding scatter visualisation |

---

## ⚙️ ML/AI Components

| Component | Model/Algorithm | Key Parameters | Library |
|---|---|---|---|
| Semantic Search | `all-MiniLM-L6-v2` | dim=384, cosine similarity, top-k=10 | `sentence-transformers`, `faiss-cpu` |
| Vector Index | FAISS `IndexFlatIP` | inner product on normalised vectors | `faiss-cpu` |
| Topic Clustering | BERTopic (HDBSCAN + UMAP) | n_topics=2–50 (tunable), min_cluster_size=dynamic | `bertopic`, `hdbscan`, `umap-learn` |
| Network Centrality | PageRank + Betweenness | damping=0.85, k=100 approx. for large graphs | `networkx` |
| Community Detection | Louvain | weight="weight" | `python-louvain` |
| LLM Summaries | `llama-3.3-70b-versatile` | temp=0.4, max_tokens=200–400 | `groq` |

---

## 🔍 Semantic Search Examples (Zero Keyword Overlap)

These queries have **no keyword overlap** with the posts they retrieve:

| Query | Result Returned | Why It's Correct |
|---|---|---|
| `"collective ownership means of production"` | Posts about socialist economic policy in r/socialism | The model recognises the conceptual meaning of collective ownership without matching any single word |
| `"الحرية والديمقراطية"` (Arabic: freedom and democracy) | Posts about civil liberties, voting rights | Multilingual embedding space maps Arabic concepts to semantically equivalent English posts |
| `"ruler who suppresses dissent"` | Posts discussing authoritarian leadership in r/politics, r/worldpolitics | Semantic relationship between "ruler suppressing dissent" and "authoritarian" captured via embedding space |

---

## 🚀 Running Locally

```bash
# Clone the repo
git clone <your-fork-url>
cd research-engineering-intern-assignment

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the dashboard
streamlit run app.py
```

The app will be at **http://localhost:8501**.

> **Note:** On first launch, click **"Build Semantic Index"** in the Search tab. This takes ~1 minute and is cached to disk for subsequent runs.

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit dashboard
├── backend/
│   ├── data_loader.py      # Reddit JSONL loading & normalisation
│   ├── embeddings.py       # sentence-transformers + FAISS semantic search
│   ├── network.py          # NetworkX graph + PageRank/Louvain
│   ├── clustering.py       # BERTopic topic clustering
│   └── llm.py              # Groq API summaries & chatbot
├── data.jsonl              # Dataset (8,799 Reddit posts)
├── cache/                  # Auto-generated FAISS index (gitignored)
├── requirements.txt
├── yugdave-prompts.md      # AI usage documentation
└── README.md
```

---

## 🧪 Edge Cases Tested

- **Empty search query** → "Please enter a query" message, no crash
- **Single-character query** → Still performs semantic search, returns results
- **Non-English query** (Arabic/Spanish) → Embedding model handles gracefully
- **Network node removal** → Graph re-renders correctly, metrics recalculated
- **Topic clusters at extremes** → n=2 gives 2 coherent clusters; n=50 doesn't crash
- **Disconnected network components** → Displayed as isolated clusters in the graph

---

*Built by Yugdave for SimPPL Research Engineering Intern Assignment.*