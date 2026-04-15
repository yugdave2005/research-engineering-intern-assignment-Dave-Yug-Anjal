"""
network.py — Build and analyse a Reddit author-subreddit co-activity network.
Node types: authors, subreddits.
Edges: author posted in subreddit (weighted by post count, score).
Metrics: PageRank, betweenness centrality, Louvain community detection.
Handles disconnected components and high-degree node removal without crashing.
"""
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict


def build_graph(df: pd.DataFrame, min_posts: int = 2) -> nx.Graph:
    """
    Build a bipartite graph: author ↔ subreddit.
    Edge weight = sum of scores of posts.
    Only include authors who have ≥ min_posts across subreddits.
    """
    # Filter prolific authors
    author_counts = df["author"].value_counts()
    active_authors = set(author_counts[author_counts >= min_posts].index)
    # Remove bots/deleted accounts
    active_authors -= {"[deleted]", "AutoModerator", "automoderator"}

    sub_df = df[df["author"].isin(active_authors)].copy()

    G = nx.Graph()

    # Add subreddit nodes
    for sub in sub_df["subreddit"].unique():
        G.add_node(sub, node_type="subreddit", label=f"r/{sub}")

    # Add author nodes and edges
    for author, grp in sub_df.groupby("author"):
        G.add_node(author, node_type="author", label=author)
        for sub, sub_grp in grp.groupby("subreddit"):
            weight = max(1, int(sub_grp["score"].sum()))
            G.add_edge(author, sub, weight=weight, post_count=len(sub_grp))

    return G


def compute_metrics(G: nx.Graph) -> dict:
    """
    Compute PageRank, betweenness centrality, and Louvain communities.
    Handles disconnected components gracefully.
    """
    metrics = {}

    # PageRank — works on disconnected graphs
    try:
        metrics["pagerank"] = nx.pagerank(G, weight="weight", max_iter=500)
    except Exception:
        metrics["pagerank"] = {n: 1 / max(len(G), 1) for n in G.nodes}

    # Betweenness — use approximation on large graphs
    try:
        if len(G) > 500:
            metrics["betweenness"] = nx.betweenness_centrality(
                G, k=min(100, len(G)), weight="weight", normalized=True
            )
        else:
            metrics["betweenness"] = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        metrics["betweenness"] = {n: 0.0 for n in G.nodes}

    # Louvain community detection
    try:
        from community import best_partition           # python-louvain
        metrics["community"] = best_partition(G, weight="weight")
    except Exception:
        # Fallback: connected-components-based labeling
        community_map = {}
        for i, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                community_map[node] = i
        metrics["community"] = community_map

    return metrics


def graph_to_df(G: nx.Graph, metrics: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert graph to node/edge DataFrames for visualization."""
    pagerank   = metrics.get("pagerank", {})
    betweenness = metrics.get("betweenness", {})
    community  = metrics.get("community", {})

    nodes = []
    for n, data in G.nodes(data=True):
        nodes.append({
            "id": n,
            "label": data.get("label", n),
            "node_type": data.get("node_type", "author"),
            "degree": G.degree(n),
            "pagerank": pagerank.get(n, 0),
            "betweenness": betweenness.get(n, 0),
            "community": community.get(n, 0),
        })

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "weight": data.get("weight", 1),
            "post_count": data.get("post_count", 1),
        })

    return pd.DataFrame(nodes), pd.DataFrame(edges)


def remove_node(G: nx.Graph, node_id: str) -> nx.Graph:
    """Return a copy of G with node_id removed. Safe for disconnected graphs."""
    G2 = G.copy()
    if node_id in G2:
        G2.remove_node(node_id)
    return G2


def get_top_nodes(G: nx.Graph, metrics: dict, n: int = 10, by: str = "pagerank") -> list[str]:
    scores = metrics.get(by, {})
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:n]
