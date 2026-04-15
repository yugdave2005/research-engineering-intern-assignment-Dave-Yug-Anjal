"""
app.py — SimPPL Political Narrative Analysis Dashboard
A deep-dive into Reddit political discourse: time-series trends, network analysis,
semantic search with LLM chatbot, and topic clustering with embedding visualization.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from pathlib import Path

# ── Page config (must be first) ─────────────────────────────────────────────
st.set_page_config(
    page_title="NarrativeScope — Reddit Political Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d1117; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #1c2128 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1c2128, #21262d);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Section headers */
.section-header {
    background: linear-gradient(135deg, #1c2128, #21262d);
    border-left: 4px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 12px 20px;
    margin: 24px 0 16px 0;
}
.section-header h2 { color: #e6edf3; margin: 0; font-size: 1.3rem; }
.section-header p { color: #8b949e; margin: 4px 0 0; font-size: 0.85rem; }

/* LLM Summary box */
.llm-summary {
    background: linear-gradient(135deg, #1a1f2e, #1e2433);
    border: 1px solid #3d4d6b;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #c9d1d9;
    font-size: 0.9rem;
    line-height: 1.6;
}
.llm-summary::before {
    content: "🤖 AI Analysis";
    display: block;
    font-size: 0.75rem;
    color: #58a6ff;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

/* Post card */
.post-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
    transition: border-color 0.2s;
}
.post-card:hover { border-color: #58a6ff; }
.post-subreddit {
    font-size: 0.75rem;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    display: inline-block;
    margin-bottom: 8px;
    font-weight: 600;
}
.post-title { color: #e6edf3; font-weight: 600; font-size: 0.95rem; margin-bottom: 6px; }
.post-meta { color: #8b949e; font-size: 0.8rem; }
.relevance-badge {
    float: right;
    background: linear-gradient(135deg, #0d4429, #1a7f4b);
    color: #3fb950;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Chat messages */
.chat-user {
    background: linear-gradient(135deg, #1f3b6e, #1a4480);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #cdd9e5;
    max-width: 80%;
    float: right;
    clear: both;
}
.chat-bot {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #c9d1d9;
    max-width: 85%;
    float: left;
    clear: both;
    line-height: 1.6;
}
.chat-clear { clear: both; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-bottom: 2px solid transparent;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
}

/* Inputs */
.stTextInput input, .stSelectbox select, .stTextArea textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
.stSlider { color: #58a6ff; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
    padding: 8px 20px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff);
    transform: translateY(-1px);
}

/* Plotly charts background */
.js-plotly-plot .plotly { background: transparent !important; }

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Loading spinner */
.loading-text { color: #58a6ff; font-style: italic; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Load data (cached) ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_df():
    from backend.data_loader import get_data
    return get_data()


@st.cache_data(show_spinner=False)
def build_embeddings(df_hash: int):
    """Build FAISS index (cached by data hash)."""
    from backend.embeddings import build_index, _load_index, IDX_FILE
    if not IDX_FILE.exists():
        from backend.data_loader import get_data
        df = get_data()
        build_index(df)
    return True


# ── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("""
    <div style='text-align:center; padding: 20px 0 16px;'>
        <div style='font-size:2.5rem;'>🔬</div>
        <div style='font-size:1.3rem; font-weight:700; color:#e6edf3;'>NarrativeScope</div>
        <div style='font-size:0.78rem; color:#8b949e; margin-top:4px;'>Reddit Political Discourse Analysis</div>
    </div>
    <hr style='border-color:#30363d; margin:0 0 16px;'>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("**🗓 Date Range**")
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input(
        "Filter by date", value=(min_date, max_date),
        min_value=min_date, max_value=max_date, label_visibility="collapsed"
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    st.sidebar.markdown("**🏛 Subreddits**")
    all_subs = sorted(df["subreddit"].unique())
    selected_subs = st.sidebar.multiselect(
        "Select subreddits", all_subs, default=all_subs,
        label_visibility="collapsed"
    )
    if not selected_subs:
        selected_subs = all_subs

    st.sidebar.markdown("**🔗 Content Type**")
    content_type = st.sidebar.radio(
        "Show", ["All posts", "External links only", "Self posts only"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style='font-size:0.78rem; color:#8b949e;'>
        📊 Dataset: <b style='color:#58a6ff;'>{len(df):,}</b> posts<br>
        📅 {min_date} → {max_date}<br>
        🏛 {len(all_subs)} subreddits
    </div>
    """, unsafe_allow_html=True)

    return start_date, end_date, selected_subs, content_type


def apply_filters(df, start_date, end_date, selected_subs, content_type):
    mask = (
        (df["date"] >= start_date) &
        (df["date"] <= end_date) &
        (df["subreddit"].isin(selected_subs))
    )
    if content_type == "External links only":
        mask &= df["is_external"]
    elif content_type == "Self posts only":
        mask &= ~df["is_external"]
    return df[mask].copy()


# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#c9d1d9"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
)

SUBREDDIT_COLORS = {
    "neoliberal": "#388bfd",
    "politics": "#58a6ff",
    "worldpolitics": "#79c0ff",
    "socialism": "#f78166",
    "Liberal": "#56d364",
    "Conservative": "#e3b341",
    "Anarchism": "#bc8cff",
    "democrats": "#2da44e",
    "Republican": "#da3633",
    "PoliticalDiscussion": "#ffa657",
}


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW & TIME SERIES
# ════════════════════════════════════════════════════════════════════════════════
def render_overview(df: pd.DataFrame, fdf: pd.DataFrame):
    from backend.llm import generate_chart_summary

    # ── KPI Cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, f"{len(fdf):,}", "Total Posts"),
        (c2, f"{fdf['author'].nunique():,}", "Unique Authors"),
        (c3, f"{fdf['subreddit'].nunique()}", "Subreddits"),
        (c4, f"{int(fdf['score'].mean()):,}", "Avg Score"),
        (c5, f"{fdf['is_external'].sum():,}", "External Links"),
    ]
    for col, val, lbl in cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Posts over time ───────────────────────────────────────────────────────
    st.markdown("""<div class="section-header">
    <h2>📈 Post Volume Over Time</h2>
    <p>Daily posting activity by subreddit</p></div>""", unsafe_allow_html=True)

    time_unit = st.radio("Group by", ["Day", "Week", "Month"], horizontal=True, key="time_unit")
    col_map = {"Day": "date", "Week": "week", "Month": "year_month"}
    tcol = col_map[time_unit]

    ts = fdf.groupby([tcol, "subreddit"]).size().reset_index(name="count")
    fig_ts = px.area(
        ts, x=tcol, y="count", color="subreddit",
        color_discrete_map=SUBREDDIT_COLORS,
        template="plotly_dark",
        labels={"count": "Posts", tcol: ""},
    )
    fig_ts.update_layout(**PLOTLY_LAYOUT)
    fig_ts.update_traces(opacity=0.8)
    st.plotly_chart(fig_ts, use_container_width=True)

    # LLM summary
    peak_date = ts.groupby(tcol)["count"].sum().idxmax()
    peak_val  = ts.groupby(tcol)["count"].sum().max()
    top_sub   = ts.groupby("subreddit")["count"].sum().idxmax()
    desc = (f"Time period covers {ts[tcol].nunique()} {time_unit.lower()}s. "
            f"Peak activity: {peak_val} posts on {peak_date}. "
            f"Most active subreddit: r/{top_sub} with {ts[ts['subreddit']==top_sub]['count'].sum()} posts total.")

    with st.spinner("🤖 Generating AI summary…"):
        summary = generate_chart_summary("time-series", desc)
    st.markdown(f'<div class="llm-summary">{summary}</div>', unsafe_allow_html=True)

    # ── Score distribution ────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""<div class="section-header">
        <h2>⭐ Score Distribution by Subreddit</h2>
        <p>Engagement (upvotes) across communities</p></div>""", unsafe_allow_html=True)

        score_data = fdf[fdf["score"] > 0]
        fig_box = px.box(
            score_data, x="subreddit", y="score",
            color="subreddit", color_discrete_map=SUBREDDIT_COLORS,
            template="plotly_dark", log_y=True,
            labels={"score": "Score (log)", "subreddit": ""},
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_r:
        st.markdown("""<div class="section-header">
        <h2>🔗 Top External Domains</h2>
        <p>Most-shared news sources across all subreddits</p></div>""", unsafe_allow_html=True)

        ext = fdf[fdf["is_external"]].copy()
        ext["domain_clean"] = ext["domain"].str.replace(r"^www\.", "", regex=True)
        top_domains = ext["domain_clean"].value_counts().head(15).reset_index()
        top_domains.columns = ["domain", "count"]
        fig_dom = px.bar(
            top_domains, x="count", y="domain", orientation="h",
            color="count", color_continuous_scale="Blues",
            template="plotly_dark",
            labels={"count": "Posts", "domain": ""},
        )
        fig_dom.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        fig_dom.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_dom, use_container_width=True)

    # LLM summary for scores
    avg_by_sub = fdf.groupby("subreddit")["score"].mean().sort_values(ascending=False)
    score_desc = (f"Average score by subreddit: {avg_by_sub.head(3).to_dict()}. "
                  f"Median post score: {int(fdf['score'].median())}. "
                  f"Posts with score > 100: {(fdf['score']>100).sum()}.")
    with st.spinner("🤖 Generating AI summary…"):
        score_summary = generate_chart_summary("score distribution", score_desc)
    st.markdown(f'<div class="llm-summary">{score_summary}</div>', unsafe_allow_html=True)

    # ── Upvote ratio heatmap ──────────────────────────────────────────────────
    st.markdown("""<div class="section-header">
    <h2>🗺 Upvote Ratio Heatmap</h2>
    <p>Consensus vs controversy: how divisive is each subreddit over time?</p></div>""",
    unsafe_allow_html=True)

    heat = fdf.groupby(["year_month", "subreddit"])["upvote_ratio"].mean().reset_index()
    heat_pivot = heat.pivot(index="subreddit", columns="year_month", values="upvote_ratio")
    fig_heat = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=heat_pivot.index.tolist(),
        colorscale="RdYlGn",
        zmin=0.4, zmax=1.0,
        colorbar=dict(title="Upvote Ratio", tickfont=dict(color="#c9d1d9")),
        hoverongaps=False,
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=350)
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — NETWORK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
def render_network(df: pd.DataFrame, fdf: pd.DataFrame):
    from backend.network import build_graph, compute_metrics, graph_to_df, remove_node, get_top_nodes
    from backend.llm import generate_network_insight

    st.markdown("""<div class="section-header">
    <h2>🕸 Author–Subreddit Network</h2>
    <p>Who posts where? PageRank influence scores, Louvain communities, betweenness centrality.</p></div>""",
    unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        min_posts = st.slider("Min posts per author", 2, 20, 3, key="net_minposts")
        metric_choice = st.selectbox("Size by", ["pagerank", "betweenness", "degree"], key="net_metric")
    with col2:
        remove_top = st.checkbox("Remove highest-degree node", key="net_remove")
        top_n = st.slider("Max nodes shown", 50, 500, 150, step=50, key="net_topn")

    with st.spinner("Building network…"):
        G = build_graph(fdf, min_posts=min_posts)
        metrics = compute_metrics(G)

        if remove_top:
            top_node = get_top_nodes(G, metrics, n=1, by="pagerank")[0]
            G = remove_node(G, top_node)
            metrics = compute_metrics(G)
            st.info(f"Removed highest-PageRank node: **{top_node}**. Network recalculated.")

        nodes_df, edges_df = graph_to_df(G, metrics)

    # Show only top_n nodes by chosen metric
    top_ids = set(nodes_df.nlargest(top_n, metric_choice)["id"].tolist())
    nodes_vis = nodes_df[nodes_df["id"].isin(top_ids)].copy()
    edges_vis  = edges_df[
        edges_df["source"].isin(top_ids) & edges_df["target"].isin(top_ids)
    ].copy()

    # ── Plotly network scatter ────────────────────────────────────────────────
    # Position with spring layout (networkx)
    import networkx as nx
    sub_G = G.subgraph(list(top_ids))
    pos = nx.spring_layout(sub_G, weight="weight", seed=42, k=0.8)

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    community_colors = px.colors.qualitative.Vivid
    for _, row in nodes_vis.iterrows():
        nid = row["id"]
        if nid not in pos:
            continue
        x, y = pos[nid]
        node_x.append(x); node_y.append(y)
        node_text.append(
            f"<b>{nid}</b><br>Type: {row['node_type']}<br>"
            f"PageRank: {row['pagerank']:.4f}<br>"
            f"Betweenness: {row['betweenness']:.4f}<br>"
            f"Community: {row['community']}<br>Degree: {row['degree']}"
        )
        node_color.append(community_colors[int(row["community"]) % len(community_colors)])
        size_val = row[metric_choice] if metric_choice in nodes_vis.columns else row["degree"]
        node_size.append(max(8, min(40, float(size_val) * 3000 if metric_choice == "pagerank" else float(size_val) * 30)))

    # Edges
    edge_x, edge_y = [], []
    for _, erow in edges_vis.iterrows():
        s, t = erow["source"], erow["target"]
        if s in pos and t in pos:
            sx, sy = pos[s]; tx, ty = pos[t]
            edge_x += [sx, tx, None]; edge_y += [sy, ty, None]

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#30363d"),
        hoverinfo="none", name="",
    ))
    fig_net.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=1, color="#0d1117")),
        text=node_text, hoverinfo="text",
        name="",
    ))
    # Build a network-specific layout (override xaxis/yaxis from PLOTLY_LAYOUT)
    net_layout = {**PLOTLY_LAYOUT}
    net_layout["xaxis"] = dict(showgrid=False, zeroline=False, showticklabels=False)
    net_layout["yaxis"] = dict(showgrid=False, zeroline=False, showticklabels=False)
    fig_net.update_layout(
        **net_layout,
        height=600,
        showlegend=False,
        title=dict(text=f"Network — {len(nodes_vis)} nodes, {len(edges_vis)} edges | "
                        f"Colored by Louvain community | Size by {metric_choice}",
                   font=dict(size=13, color="#8b949e")),
    )
    st.plotly_chart(fig_net, use_container_width=True)

    # ── Top authors table ─────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**🏆 Top Authors by PageRank**")
        top_authors = nodes_df[nodes_df["node_type"] == "author"]\
            .nlargest(10, "pagerank")[["id","degree","pagerank","betweenness","community"]]\
            .rename(columns={"id":"Author","degree":"Degree","pagerank":"PageRank",
                             "betweenness":"Betweenness","community":"Community"})
        top_authors["PageRank"] = top_authors["PageRank"].map("{:.5f}".format)
        top_authors["Betweenness"] = top_authors["Betweenness"].map("{:.4f}".format)
        st.dataframe(top_authors, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("**🏛 Subreddit Centrality**")
        sub_nodes = nodes_df[nodes_df["node_type"] == "subreddit"]\
            .nlargest(10, "pagerank")[["id","degree","pagerank"]]\
            .rename(columns={"id":"Subreddit","degree":"Authors","pagerank":"PageRank"})
        sub_nodes["PageRank"] = sub_nodes["PageRank"].map("{:.5f}".format)
        st.dataframe(sub_nodes, use_container_width=True, hide_index=True)

    # LLM insight
    with st.spinner("🤖 Generating network insight…"):
        top_auth_ids = nodes_df[nodes_df["node_type"]=="author"].nlargest(5,"pagerank")["id"].tolist()
        top_sub_ids  = nodes_df[nodes_df["node_type"]=="subreddit"].nlargest(5,"degree")["id"].tolist()
        insight = generate_network_insight(top_auth_ids, top_sub_ids, len(edges_df))
    st.markdown(f'<div class="llm-summary">{insight}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEMANTIC SEARCH & CHATBOT
# ════════════════════════════════════════════════════════════════════════════════
def render_search(df: pd.DataFrame, fdf: pd.DataFrame):
    from backend.embeddings import semantic_search
    from backend.llm import chatbot_response

    st.markdown("""<div class="section-header">
    <h2>🔍 Semantic Search & AI Chatbot</h2>
    <p>Search by meaning, not just keywords. Works in any language. Ask follow-up questions.</p>
    </div>""", unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = pd.DataFrame()

    # Build index button
    cache_ok = False
    try:
        from backend.embeddings import IDX_FILE
        cache_ok = IDX_FILE.exists()
    except Exception:
        pass

    if not cache_ok:
        st.warning("⚡ Semantic search index not built yet. Click below to build it (takes ~1 min).")
        if st.button("Build Semantic Index"):
            with st.spinner("Building embedding index for all posts…"):
                from backend.embeddings import build_index
                build_index(df, text_col="full_text")
            st.success("✅ Index built! You can now search semantically.")
            st.rerun()
        return

    # ── Search bar ────────────────────────────────────────────────────────────
    col_q, col_k = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Ask anything about the dataset…",
            placeholder="e.g. 'authoritarian policies', 'environmental policy debate', 'حرية الصحافة'",
            label_visibility="collapsed",
            key="search_query"
        )
    with col_k:
        top_k = st.selectbox("Results", [5, 10, 20], index=1, label_visibility="collapsed")

    if st.button("🔍 Search", key="search_btn") or query:
        if not query or not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching semantically…"):
                results = semantic_search(query, df, top_k=top_k)  # search full df always
                st.session_state.search_results = results

            if results.empty:
                st.error("No results found. Try a different query or broaden your filters.")
            else:
                # Show posts
                st.markdown(f"**Found {len(results)} semantically relevant posts** for: *{query}*")

                for _, row in results.iterrows():
                    score_pct = int(row.get("relevance_score", 0) * 100)
                    text_preview = str(row.get("selftext", ""))[:200].strip()
                    if text_preview and text_preview != "[removed]":
                        preview_html = f"<br><small style='color:#8b949e'>{text_preview}…</small>"
                    else:
                        preview_html = ""

                    st.markdown(f"""
                    <div class="post-card">
                        <span class="post-subreddit">r/{row.get('subreddit','?')}</span>
                        <span class="relevance-badge">Match: {score_pct}%</span>
                        <div class="post-title">{row.get('title', 'No title')}</div>
                        {preview_html}
                        <div class="post-meta">
                            👤 u/{row.get('author','?')} &nbsp;·&nbsp;
                            ⭐ {int(row.get('score',0))} &nbsp;·&nbsp;
                            💬 {int(row.get('num_comments',0))} comments &nbsp;·&nbsp;
                            🔗 {row.get('domain','self')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Chatbot answer
                st.markdown("---")
                st.markdown("**🤖 AI Analysis of Results**")
                with st.spinner("Generating AI analysis…"):
                    posts_ctx = results.head(5).to_dict("records")
                    answer = chatbot_response(query, posts_ctx)

                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "bot", "content": answer})

    # ── Chat history display ──────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("**💬 Conversation History**")
        for msg in st.session_state.chat_history[-10:]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🗣 {msg["content"]}</div><div class="chat-clear"></div>',
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">{msg["content"]}</div><div class="chat-clear"></div>',
                           unsafe_allow_html=True)
        if st.button("🗑 Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Example queries ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**💡 Try these example queries** (zero keyword overlap with expected results):")
    examples = [
        ("\"Authoritarianism rising\"", "Find posts about erosion of democratic institutions"),
        ("\"الحرية والديمقراطية\"", "Arabic query: finds posts about freedom and democracy"),
        ("\"collective ownership means of production\"", "Finds posts about socialism/communism policy"),
    ]
    for query_text, note in examples:
        st.markdown(f"- `{query_text}` — *{note}*")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — TOPIC CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════
def render_clustering(df: pd.DataFrame, fdf: pd.DataFrame):
    from backend.clustering import cluster_topics, get_topic_summary
    from backend.llm import generate_chart_summary

    st.markdown("""<div class="section-header">
    <h2>🧩 Topic Clustering & Embedding Visualization</h2>
    <p>BERTopic clusters posts by semantic topic. UMAP 2D projection shows how posts cluster in embedding space.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        n_topics = st.slider(
            "Number of topics", min_value=2, max_value=50, value=10, step=1,
            help="Controls how many topics BERTopic aims to find. Extremes (2 or 50) are handled gracefully."
        )
        sample_size = st.slider(
            "Sample size", min_value=500, max_value=min(len(fdf), 5000),
            value=min(2000, len(fdf)), step=500,
            help="Clustering on a subset for speed. Full dataset recommended for production."
        )
        run_clustering = st.button("🔄 Run Clustering", key="cluster_btn", type="primary")

    with col2:
        st.info("💡 Clustering uses BERTopic (HDBSCAN + UMAP). The slider controls target topic count. "
                "Some posts may be 'Miscellaneous' (outliers not assigned to any cluster).")

    if run_clustering or "cluster_result" not in st.session_state:
        # Get sample indices
        sample_size = min(sample_size, len(fdf))
        sample_indices = fdf.sample(sample_size, random_state=42).index
        sample_df = fdf.loc[sample_indices].reset_index(drop=True)

        # Try to load pre-computed embeddings for the sample to speed up clustering
        sample_embs = None
        emb_path = Path("cache/embeddings.npy")
        if emb_path.exists():
            try:
                all_embs = np.load(str(emb_path))
                # Map fdf indices back to the original df positions (assumes fdf is from df)
                # We need the positional indices in the original df to match embeddings.npy
                sample_embs = all_embs[sample_indices]
            except Exception as e:
                print(f"Failed to load cached embeddings: {e}")

        with st.spinner(f"🔬 Clustering {len(sample_df)} posts into ~{n_topics} topics…"):
            try:
                # result_df now includes umap_x and umap_y
                result_df, model = cluster_topics(sample_df, n_topics=n_topics, embeddings=sample_embs)
                topic_summary   = get_topic_summary(model)
                st.session_state["cluster_result"] = result_df
                st.session_state["cluster_summary"] = topic_summary
                st.session_state["cluster_model"]   = model
                st.success(f"✅ Found {topic_summary[topic_summary['topic_id'] != -1]['topic_id'].nunique()} topics!")
            except Exception as e:
                st.error(f"Clustering failed: {e}")
                st.exception(e)
                return

    result_df    = st.session_state.get("cluster_result", pd.DataFrame())
    topic_summary = st.session_state.get("cluster_summary", pd.DataFrame())

    if result_df.empty:
        st.warning("Run clustering first.")
        return

    # ── UMAP scatter ──────────────────────────────────────────────────────────
    st.markdown("**📐 UMAP Embedding Space** — Each point is a post, colored by topic")
    plot_df = result_df[result_df["topic_id"] != -1].copy()
    plot_df["topic_label_short"] = plot_df["topic_label"].str[:30]

    fig_umap = px.scatter(
        plot_df, x="umap_x", y="umap_y",
        color="topic_label_short",
        hover_data={"title": True, "subreddit": True, "score": True,
                    "umap_x": False, "umap_y": False},
        template="plotly_dark",
        opacity=0.7,
        size_max=6,
        labels={"topic_label_short": "Topic", "umap_x": "", "umap_y": ""},
    )
    fig_umap.update_traces(marker=dict(size=5))
    cl_layout = {**PLOTLY_LAYOUT}
    cl_layout["legend"] = dict(itemsizing="constant", font=dict(size=11))
    fig_umap.update_layout(**cl_layout, height=550)
    st.plotly_chart(fig_umap, use_container_width=True)

    # ── Topic table ──────────────────────────────────────────────────────────-
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**📋 Topic Summary**")
        display_topics = topic_summary[topic_summary["topic_id"] != -1]\
            .rename(columns={"label":"Topic","size":"Posts","top_words":"Top Keywords"})\
            [["Topic","Posts","Top Keywords"]]
        st.dataframe(display_topics, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("**📊 Topic Size Distribution**")
        fig_bar = px.bar(
            topic_summary[topic_summary["topic_id"] != -1].head(15),
            x="size", y="label", orientation="h",
            color="size", color_continuous_scale="Purples",
            template="plotly_dark",
            labels={"size":"Posts","label":"Topic"},
        )
        fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=400)
        fig_bar.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_bar, use_container_width=True)

    # LLM summary
    top3 = topic_summary[topic_summary["topic_id"] != -1].head(3)
    desc = (f"Clustering found {len(topic_summary)-1} topics. "
            f"Top topics: {top3['label'].tolist()} with {top3['size'].tolist()} posts respectively.")
    from backend.llm import generate_chart_summary
    with st.spinner("🤖 Generating cluster analysis…"):
        cluster_sum = generate_chart_summary("topic clustering scatter", desc)
    st.markdown(f'<div class="llm-summary">{cluster_sum}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main():
    # Load data
    with st.spinner("📦 Loading dataset…"):
        df = load_df()

    # Sidebar filters
    start_date, end_date, selected_subs, content_type = render_sidebar(df)
    fdf = apply_filters(df, start_date, end_date, selected_subs, content_type)

    if fdf.empty:
        st.error("No posts match your current filters. Adjust the sidebar settings.")
        return

    # Hero header
    st.markdown("""
    <div style='padding: 32px 0 8px; text-align: center;'>
        <h1 style='font-size:2.4rem; font-weight:800;
            background: linear-gradient(135deg, #58a6ff, #bc8cff, #f78166);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            margin-bottom: 6px;'>
            🔬 NarrativeScope
        </h1>
        <p style='color:#8b949e; font-size:1rem; margin:0;'>
            Reddit Political Discourse Analysis &nbsp;·&nbsp; SimPPL Research Engineering Assignment
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview & Trends",
        "🕸 Network Analysis",
        "🔍 Semantic Search",
        "🧩 Topic Clusters",
    ])

    with tab1:
        render_overview(df, fdf)
    with tab2:
        render_network(df, fdf)
    with tab3:
        render_search(df, fdf)
    with tab4:
        render_clustering(df, fdf)


if __name__ == "__main__":
    main()
