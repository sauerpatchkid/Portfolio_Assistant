# --- Web Search Tool ---
# Free web search via DuckDuckGo (no API key). Use this for general market news,
# central-bank decisions, macro events, or anything outside our DB / yfinance coverage.
from agent.market import snapshot_active


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web via DuckDuckGo. Returns top result titles, snippets, and URLs."""
    if not query or not query.strip():
        return {"error": "Empty search query."}

    # Live web results can't be scored or reproduced, so offline runs skip them.
    if snapshot_active():
        return {"error": "Web search is unavailable in offline evaluation mode."}

    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return {"error": f"Web search failed: {e}"}

    if not raw_results:
        return {"query": query, "num_results": 0, "results": []}

    results = [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", ""),
        }
        for r in raw_results
    ]

    return {"query": query, "num_results": len(results), "results": results}
