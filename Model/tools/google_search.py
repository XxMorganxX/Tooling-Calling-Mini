"""
Google Search Tool – standalone module.

Ported from MCP BaseTool class to module-level functions.
Provides Google-like search with AI Overview or Gemini-powered summaries.
"""

import logging
import os
import re
import sys
import time
from typing import Any, Dict, Optional
from urllib.parse import quote

from tools.models import GoogleSearchArgs

try:
    from clients.web_search_client import Websearch
except ImportError:
    Websearch = None

try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except ImportError:
    genai = None
    _HAS_GEMINI = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (lazy-init)
# ---------------------------------------------------------------------------

_searcher: Optional[Any] = None
_searcher_init_attempted = False
_init_error: Optional[str] = None
_gemini_model: Optional[Any] = None

# Pattern constants
LINK_PATTERNS = [
    r"\blink\s+to\b", r"\burl\s+(for|to)\b", r"\bwebsite\s+(for|of)\b",
    r"\bfind\s+(me\s+)?(the\s+)?link\b", r"\bgive\s+(me\s+)?(the\s+)?link\b",
    r"\bget\s+(me\s+)?(the\s+)?link\b", r"\bwhere\s+can\s+i\s+find\b",
]
DIRECTIONS_PATTERNS = [
    r"\bdirections?\s+to\b", r"\bhow\s+(do\s+i\s+)?get\s+to\b",
    r"\broute\s+to\b", r"\bnavigate\s+to\b", r"\bdriving\s+to\b",
    r"\bwalking\s+to\b", r"\btransit\s+to\b",
]


def _log_timing(component: str, elapsed_ms: float, extra: str = "") -> None:
    extra_str = f" | {extra}" if extra else ""
    print(f"[{component}] {elapsed_ms:.0f}ms{extra_str}", file=sys.stderr)


def _ensure_initialized() -> None:
    """Lazy-init searcher and Gemini model."""
    global _searcher, _searcher_init_attempted, _init_error, _gemini_model

    if _searcher_init_attempted:
        return
    _searcher_init_attempted = True

    if Websearch is None:
        _init_error = "Websearch client not available (missing serpapi package)"
        logger.warning(_init_error)
    else:
        try:
            _searcher = Websearch()
            if not _searcher.api_key:
                _init_error = "SERPAPI_API_KEY not configured in environment"
                logger.warning(_init_error)
        except Exception as e:
            _init_error = f"Failed to initialize Websearch: {e}"
            logger.error(_init_error)

    if _HAS_GEMINI:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest"]:
                    try:
                        _gemini_model = genai.GenerativeModel(model_name)
                        break
                    except Exception:
                        continue
            except Exception as e:
                logger.warning("Failed to initialize Gemini: %s", e)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute(args: GoogleSearchArgs) -> dict:
    """Execute a Google-style search."""
    _ensure_initialized()
    total_start = time.perf_counter()

    try:
        query = args.query.strip() if args.query else ""
        query_type = (args.query_type or "").strip().lower()

        if not query:
            return {
                "result": "",
                "source": "",
                "success": False,
                "error": "Search query is empty. Provide a non-empty 'query' string to search for",
            }

        if not query_type or query_type not in ["general", "link", "directions"]:
            query_type = _detect_query_type(query)

        if _searcher is None:
            return {
                "result": "",
                "source": "",
                "success": False,
                "error": _init_error or "Search client not initialized",
                "suggestion": "Install serpapi: pip install google-search-results, and set SERPAPI_API_KEY",
            }

        # Directions → instant maps link
        if query_type == "directions":
            destination = _extract_destination(query)
            maps_link = _generate_maps_link(destination)
            total_elapsed = (time.perf_counter() - total_start) * 1000
            _log_timing("TOTAL (directions)", total_elapsed, "instant maps link")
            return {
                "result": maps_link,
                "source": "Google Maps",
                "success": True,
                "query_type": "directions",
                "destination": destination,
            }

        # Link query
        if query_type == "link":
            results = _searcher.get_search_results(query, num_results=3)
            if results:
                best_link = _succinct_summarize(query, results, "link")
                if best_link and best_link.startswith("http"):
                    total_elapsed = (time.perf_counter() - total_start) * 1000
                    _log_timing("TOTAL (link query)", total_elapsed, "Gemini picked link")
                    return {"result": best_link, "source": "Search", "success": True, "query_type": "link"}
                top_result = results[0]
                total_elapsed = (time.perf_counter() - total_start) * 1000
                _log_timing("TOTAL (link query)", total_elapsed, "first result fallback")
                return {
                    "result": top_result.get("link", ""),
                    "source": "Search",
                    "success": True,
                    "query_type": "link",
                    "title": top_result.get("title", ""),
                }
            total_elapsed = (time.perf_counter() - total_start) * 1000
            _log_timing("TOTAL (link query)", total_elapsed, "no results")
            return {"result": "No relevant link found.", "source": "Search", "success": False}

        # General query: AI Overview first
        ai = _searcher.get_ai_overview(query)
        ai_text = _searcher.flatten_ai_overview(ai)

        if ai_text:
            total_elapsed = (time.perf_counter() - total_start) * 1000
            _log_timing("TOTAL (general)", total_elapsed, "SUCCESS via AI Overview")
            return {"result": ai_text, "source": "AI Overview", "success": True, "query_type": "general"}

        # Fallback: organic results + summarize
        results = _searcher.get_search_results(query, num_results=8)

        if not results:
            total_elapsed = (time.perf_counter() - total_start) * 1000
            _log_timing("TOTAL (general)", total_elapsed, "no results found")
            return {"result": "No search results found for this query.", "source": "Search", "success": False}

        # Try succinct summarization
        summary = _succinct_summarize(query, results, "general")
        if summary:
            total_elapsed = (time.perf_counter() - total_start) * 1000
            _log_timing("TOTAL (general)", total_elapsed, "SUCCESS via Gemini Succinct")
            return {"result": summary, "source": "Gemini", "success": True, "query_type": "general"}

        # Standard Gemini summarization
        summary = _searcher._summarize_with_gemini(query, results)
        if summary:
            total_elapsed = (time.perf_counter() - total_start) * 1000
            _log_timing("TOTAL (general)", total_elapsed, "SUCCESS via Gemini Standard")
            return {"result": summary, "source": "Gemini", "success": True, "query_type": "general"}

        # Last resort: raw results
        formatted = _searcher.format_results(results)
        if formatted:
            total_elapsed = (time.perf_counter() - total_start) * 1000
            _log_timing("TOTAL (general)", total_elapsed, "SUCCESS via raw results")
            return {"result": formatted, "source": "Search results", "success": True}

        total_elapsed = (time.perf_counter() - total_start) * 1000
        _log_timing("TOTAL (general)", total_elapsed, "all methods failed")
        return {"result": "Search completed but no summary available.", "source": "Search", "success": False}

    except Exception as e:
        total_elapsed = (time.perf_counter() - total_start) * 1000
        _log_timing("TOTAL", total_elapsed, f"EXCEPTION: {e}")
        logger.error("Google search failed: %s", e)
        return {"result": f"Search failed: {e}", "source": "", "success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Query type detection
# ---------------------------------------------------------------------------


def _detect_query_type(query: str) -> str:
    query_lower = query.lower()
    for pattern in DIRECTIONS_PATTERNS:
        if re.search(pattern, query_lower):
            return "directions"
    for pattern in LINK_PATTERNS:
        if re.search(pattern, query_lower):
            return "link"
    return "general"


def _extract_destination(query: str) -> str:
    query_lower = query.lower()
    for pattern in DIRECTIONS_PATTERNS + LINK_PATTERNS:
        query_lower = re.sub(pattern, "", query_lower)
    destination = query_lower.strip()
    destination = re.sub(r"^(the\s+)?", "", destination)
    return destination.strip() or query


def _generate_maps_link(destination: str) -> str:
    encoded = quote(destination)
    return f"https://www.google.com/maps/dir/?api=1&destination={encoded}"


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def _succinct_summarize(query: str, results: list, query_type: str) -> Optional[str]:
    """Generate a succinct summary using Gemini with query-type-specific prompts."""
    t_start = time.perf_counter()

    if not _gemini_model or not results:
        _log_timing("Gemini Succinct Summary", 0, "SKIPPED (no model or results)")
        return None

    try:
        items = []
        for r in results[:5]:
            title = (r.get("title") or "").strip()
            link = (r.get("link") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            if title or link:
                items.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")

        sources_block = "\n\n".join(items)

        if query_type == "directions":
            prompt = (
                f"User wants directions to a location. From the search results below, "
                f"provide ONLY a Google Maps link. If a maps link exists in the results, use it. "
                f"Otherwise, say 'Use Google Maps: https://www.google.com/maps/dir/?api=1&destination=[encoded address]' "
                f"with the actual destination encoded.\n\n"
                f"Query: {query}\n\nSources:\n{sources_block}\n\n"
                f"Return ONLY the link, nothing else."
            )
        elif query_type == "link":
            prompt = (
                f"User wants a specific link/URL. From the search results below, "
                f"identify the most relevant URL and return ONLY that URL. "
                f"No explanation, just the link.\n\n"
                f"Query: {query}\n\nSources:\n{sources_block}\n\n"
                f"Return ONLY the most relevant URL."
            )
        else:
            prompt = (
                f"Answer the user's query in 1-2 sentences max. Be direct and factual. "
                f"No preamble like 'Based on...' or 'According to...'. Just the answer.\n\n"
                f"Query: {query}\n\nSources:\n{sources_block}"
            )

        gen_start = time.perf_counter()
        resp = _gemini_model.generate_content(prompt)
        gen_elapsed = (time.perf_counter() - gen_start) * 1000

        text = getattr(resp, "text", None)
        total_elapsed = (time.perf_counter() - t_start) * 1000

        if text:
            _log_timing("Gemini Succinct Summary", total_elapsed, f"type={query_type} (generation: {gen_elapsed:.0f}ms)")
            return text.strip()
        else:
            _log_timing("Gemini Succinct Summary", total_elapsed, "FAILED (empty response)")
    except Exception as e:
        total_elapsed = (time.perf_counter() - t_start) * 1000
        _log_timing("Gemini Succinct Summary", total_elapsed, f"FAILED: {e}")
        logger.warning("Gemini summarization failed: %s", e)

    return None
