import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from app.retrieval import retrieve

load_dotenv()  # make GROQ_API_KEY / TAVILY_API_KEY available to the tools


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal company and product document knowledge base.

    Use this for questions answerable by the organization's own documents:
    company policies (vacation, expenses, security, equipment) and product specs
    (the NW-3000 router, its error codes, warranty). Prefer this over web search
    for anything internal or company-specific.

    Args:
        query: what to look up in the internal documents.
    Returns the most relevant passages, each tagged with its source document.
    """
    hits = retrieve(query, k=4)
    if not hits:
        return "No relevant passages found in the knowledge base."
    return "\n\n".join(
        f"[{h.metadata.get('source', '?')}] {h.page_content.strip()}" for h in hits
    )


_tavily = None


def _get_tavily():
    """Build the Tavily search tool once, on first use."""
    global _tavily
    if _tavily is None:
        from langchain_tavily import TavilySearch
        _tavily = TavilySearch(max_results=5)   # reads TAVILY_API_KEY from env
    return _tavily


@tool
def web_search(query: str) -> str:
    """Search the live web for current, real-world, or external information.

    Use this for anything NOT in the internal documents: current events, general
    knowledge, prices, weather, public facts, or anything needing up-to-date info.
    Do NOT use this for internal company policies or NW-3000 product details —
    use search_knowledge_base for those.

    Args:
        query: a natural-language web search query.
    Returns a few relevant results (title, snippet, URL), or an error message.
    """
    try:
        result = _get_tavily().invoke({"query": query})
    except Exception as e:
        return f"error: web search failed ({e})"    # return, don't raise

    if isinstance(result, str):
        return result[:1500]
    results = result.get("results", []) if isinstance(result, dict) else []
    if not results:
        return "No web results found."

    lines = []
    for r in results[:5]:
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = (r.get("content", "") or "").strip().replace("\n", " ")
        lines.append(f"- {title} ({url})\n  {snippet[:300]}")
    return "\n".join(lines)
