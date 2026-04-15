"""
llm.py — Groq API integration for dynamic GenAI summaries and chatbot.
Falls back gracefully if API key is missing or rate-limited.
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_PRIMARY_KEY = os.getenv("GROQ_API_KEY")
_BACKUP_KEY  = os.getenv("GROQ_API_KEY_2")
MODEL = "llama-3.3-70b-versatile"


def _get_client(key: str) -> Groq:
    return Groq(api_key=key)


def _call(prompt: str, system: str = "", max_tokens: int = 300) -> str:
    """Try primary key, then backup, then return empty string on failure."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for key in [_PRIMARY_KEY, _BACKUP_KEY]:
        try:
            client = _get_client(key)
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[llm] Key failed: {e}")
    return ""


# ── Chart Summary ─────────────────────────────────────────────────────────────
def generate_chart_summary(chart_type: str, data_description: str) -> str:
    """
    Generate a plain-language 2-3 sentence summary of a chart for non-technical readers.
    data_description should be a concise, data-grounded string.
    """
    prompt = f"""You are a data journalist writing captions for a social media analysis dashboard.
Write a 2-3 sentence plain-language summary explaining the following {chart_type} chart to a non-technical reader.
Focus on the trend, peak, or notable pattern. Do not use jargon.

Data context: {data_description}

Summary:"""
    result = _call(prompt, max_tokens=200)
    if not result:
        return f"This {chart_type} shows patterns in Reddit post activity based on the selected filters."
    return result


# ── Chatbot ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert social media analyst assistant for a Reddit political narrative analysis dashboard.
You help researchers understand trends, rhetoric, and information spread across political subreddits.
When given post excerpts, answer the user's question thoughtfully and concisely (3-5 sentences).
At the end of your response, always suggest 2-3 follow-up queries the user might explore next, formatted as:
**Suggested follow-ups:**
- [query 1]
- [query 2]
- [query 3]"""


def chatbot_response(user_query: str, context_posts: list[dict]) -> str:
    """
    Generate a chatbot response given a user query and semantically retrieved post context.
    """
    if not user_query or not user_query.strip():
        return "Please enter a question or search query to get started."

    if not context_posts:
        return (
            "No relevant posts were found for your query. Try a different search term or broaden your filters.\n\n"
            "**Suggested follow-ups:**\n- What narratives are most common across all subreddits?\n"
            "- Which subreddits post the most external links?\n- Who are the most active authors in the dataset?"
        )

    context_str = "\n\n".join(
        f"[{p.get('subreddit', '?')}] {p.get('title', '')} — score: {p.get('score', 0)}"
        for p in context_posts[:5]
    )

    prompt = f"""User query: "{user_query}"

Relevant Reddit posts from the dataset:
{context_str}

Answer the user's question based on these posts and your broader analysis knowledge."""

    result = _call(prompt, system=SYSTEM_PROMPT, max_tokens=400)
    if not result:
        return (
            f"Found {len(context_posts)} semantically relevant posts for your query. "
            "The posts span multiple political communities and suggest varied perspectives on this topic.\n\n"
            "**Suggested follow-ups:**\n- What is the sentiment across these posts?\n"
            "- Which subreddit discusses this most?\n- Are there any external news links in these posts?"
        )
    return result


# ── Network Insight ────────────────────────────────────────────────────────────
def generate_network_insight(top_authors: list, top_subreddits: list, edge_count: int) -> str:
    prompt = f"""Describe in 2-3 sentences the key structural patterns in a Reddit cross-posting network.
Top authors by activity: {top_authors[:5]}
Most connected subreddits: {top_subreddits[:5]}
Total connections: {edge_count}
Focus on what this means for information flow and influence."""
    result = _call(prompt, max_tokens=200)
    return result or "The network reveals how content and users interconnect across political subreddits."
