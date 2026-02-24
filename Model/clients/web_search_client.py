"""
Simple Google AI Overview search using SerpAPI, wrapped in a class.
"""

# pip install google-search-results
import os
import sys
import time
from dotenv import load_dotenv
from serpapi import GoogleSearch
import requests


def _log_timing(component: str, elapsed_ms: float, extra: str = ""):
    """Log timing information for search components."""
    extra_str = f" | {extra}" if extra else ""
    print(f"⏱️  [{component}] {elapsed_ms:.0f}ms{extra_str}", file=sys.stderr)




class Websearch:
    """Encapsulates SerpAPI Google search with AI Overview support."""

    def __init__(self, api_key: str | None = None, location: str | None = None):
        load_dotenv()
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")
        self.location = location

    def get_ai_overview(self, query: str, *, hl: str = "en", gl: str = "us", location: str | None = None) -> dict | None:
        """Run Google search and return AI Overview payload if available.
        
        OPTIMIZED: Tries direct AI Overview engine first (single call), 
        then falls back to Google Search + page_token if needed.
        """
        total_start = time.perf_counter()
        loc = location or self.location
        ai = None
        
        # === ATTEMPT 1: Direct AI Overview request (fastest path) ===
        try:
            direct_params = {
                "engine": "google_ai_overview",
                "q": query,
                "hl": hl,
                "gl": gl,
                "api_key": self.api_key,
            }
            if loc:
                direct_params["location"] = loc
            
            t1_start = time.perf_counter()
            result = GoogleSearch(direct_params).get_dict()
            t1_elapsed = (time.perf_counter() - t1_start) * 1000
            
            ai = result.get("ai_overview")
            has_text = bool(ai.get("text_blocks")) if isinstance(ai, dict) else False
            _log_timing("AI Overview - Direct Request", t1_elapsed, f"has_text_blocks={has_text}")
            
            if ai and has_text:
                total_elapsed = (time.perf_counter() - total_start) * 1000
                _log_timing("AI Overview - Content Found", total_elapsed, "from direct request ✨")
                return ai
                
        except Exception as e:
            _log_timing("AI Overview - Direct Request", 0, f"FAILED: {e}")
        
        # === ATTEMPT 2: Fall back to Google Search with ai_overview=true ===
        try:
            search_params = {
                "engine": "google",
                "q": query,
                "hl": hl,
                "gl": gl,
                "google_domain": "google.com",
                "device": "desktop",
                "safe": "off",
                "ai_overview": "true",
                "no_cache": "true",
                "api_key": self.api_key,
            }
            if loc:
                search_params["location"] = loc
            
            t2_start = time.perf_counter()
            s2 = GoogleSearch(search_params).get_dict()
            t2_elapsed = (time.perf_counter() - t2_start) * 1000
            _log_timing("AI Overview - Google Search Fallback", t2_elapsed, "ai_overview=true")
            
            ai = (
                s2.get("ai_overview")
                or s2.get("sg_results")
                or s2.get("sge_summary")
            )
            
            has_text = bool(ai.get("text_blocks")) if isinstance(ai, dict) else False
            
            if ai and has_text:
                total_elapsed = (time.perf_counter() - total_start) * 1000
                _log_timing("AI Overview - Content Found", total_elapsed, "from Google Search fallback")
                return ai
            
            if ai and not has_text:
                token = ai.get("page_token") or s2.get("search_metadata", {}).get("ai_overview_page_token")
                if token:
                    _log_timing("AI Overview - Google Search", 0, f"got placeholder, using page_token")
                    
                    t3_start = time.perf_counter()
                    s3 = GoogleSearch({
                        "engine": "google_ai_overview",
                        "page_token": token,
                        "api_key": self.api_key,
                    }).get_dict()
                    t3_elapsed = (time.perf_counter() - t3_start) * 1000
                    _log_timing("AI Overview - Page Token Request", t3_elapsed)
                    
                    ai = s3.get("ai_overview")
                    has_text = bool(ai.get("text_blocks")) if isinstance(ai, dict) else False
                    
                    if ai and has_text:
                        total_elapsed = (time.perf_counter() - total_start) * 1000
                        _log_timing("AI Overview - Content Found", total_elapsed, "from page_token request")
                        return ai
                        
        except Exception as e:
            _log_timing("AI Overview - Google Search Fallback", 0, f"FAILED: {e}")
        
        total_elapsed = (time.perf_counter() - total_start) * 1000
        _log_timing("AI Overview - NOT FOUND", total_elapsed, "no usable content")
        return None

    def get_search_results(self, 
                           query: str, 
                           *, 
                           hl: str = "en", 
                           gl: str = "us", 
                           location: str | None = None,
                           num_results: int = 5) -> list[dict]:
        """Fetch standard Google search results (organic) as a fallback.
        Returns a list of dicts: {title, link, snippet}.
        """
        t_start = time.perf_counter()
        
        params = {
            "engine": "google",
            "q": query,
            "hl": hl,
            "gl": gl,
            "google_domain": "google.com",
            "device": "desktop",
            "safe": "off",
            "no_cache": "true",
            "num": max(1, min(int(num_results or 5), 20)),
            "api_key": self.api_key,
        }
        loc = location or self.location
        if loc:
            params["location"] = loc
        data = GoogleSearch(params).get_dict()
        
        api_elapsed = (time.perf_counter() - t_start) * 1000
        
        organic = data.get("organic_results", []) or []
        results: list[dict] = []
        for r in organic[:num_results]:
            if not isinstance(r, dict):
                continue
            title = (r.get("title") or "").strip()
            link = (r.get("link") or "").strip()
            snippet = (r.get("snippet") or r.get("snippet_highlighted_words") or "")
            if isinstance(snippet, list):
                snippet = " ".join(w for w in snippet if isinstance(w, str))
            snippet = snippet.strip() if isinstance(snippet, str) else ""
            if title or link or snippet:
                results.append({"title": title, "link": link, "snippet": snippet})
        
        total_elapsed = (time.perf_counter() - t_start) * 1000
        _log_timing("Organic Search Results", total_elapsed, f"found {len(results)} results (API: {api_elapsed:.0f}ms)")
        return results

    @staticmethod
    def flatten_ai_overview(ai_overview: dict | None) -> str | None:
        """Flatten AI Overview blocks into markdown-like text."""
        if not ai_overview:
            return None
        parts: list[str] = []
        for block in ai_overview.get("text_blocks", []) or []:
            try:
                btype = block.get("type", "").lower()
                content = (
                    block.get("snippet")
                    or block.get("text")
                    or block.get("title")
                    or ""
                )
                content = content.strip() if isinstance(content, str) else ""

                if btype == "heading":
                    if content:
                        parts.append(f"## {content}")
                elif btype == "paragraph":
                    if content:
                        parts.append(content)
                elif btype == "list":
                    items = block.get("list") or block.get("items") or []
                    for item in items:
                        ititle = (item.get("title") or "").strip() if isinstance(item, dict) else ""
                        isnippet = (
                            (item.get("snippet") if isinstance(item, dict) else None)
                            or (item.get("text") if isinstance(item, dict) else None)
                            or ""
                        )
                        isnippet = isnippet.strip() if isinstance(isnippet, str) else ""
                        if ititle or isnippet:
                            bullet = f"- **{ititle}** {isnippet}".strip() if ititle else f"- {isnippet}"
                            parts.append(bullet)
                else:
                    if content:
                        parts.append(content)
            except Exception:
                continue
        return "\n".join(parts) if parts else None

    @staticmethod
    def format_results(results: list[dict]) -> str | None:
        """Format organic results into a readable text block."""
        if not results:
            return None
        lines: list[str] = []
        for r in results:
            title = r.get("title") or ""
            link = r.get("link") or ""
            snippet = r.get("snippet") or ""
            line = f"- {title} ({link})\n  {snippet}".strip()
            lines.append(line)
        return "\n".join(lines)

    def _summarize_with_gemini(self, query: str, results: list[dict]) -> str | None:
        """Summarize search results using Gemini 1.5 Flash, if available.
        Requires GOOGLE_API_KEY and google-generativeai package.
        """
        t_start = time.perf_counter()
        try:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                _log_timing("Gemini Standard Summary", 0, "SKIPPED (no API key)")
                return None
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            model_names = [
                "gemini-2.5-flash",
                "gemini-1.5-flash",
                "gemini-1.5-flash-latest",
            ]
            model = None
            model_used = None
            for name in model_names:
                try:
                    model = genai.GenerativeModel(name)
                    model_used = name
                    break
                except Exception:
                    model = None
            if model is None:
                _log_timing("Gemini Standard Summary", 0, "SKIPPED (no model available)")
                return None

            items = []
            for r in results:
                title = (r.get("title") or "").strip()
                link = (r.get("link") or "").strip()
                snippet = (r.get("snippet") or "").strip()
                if title or link or snippet:
                    items.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")
            sources_block = "\n\n".join(items)

            prompt = (
                "You are a concise assistant. Using only the sources below, answer the user's query in 3–5 sentences. "
                "Be factual and neutral. After the answer, add a short 'Sources:' section listing 2–3 of the most relevant URLs.\n\n"
                f"User query: {query}\n\n"
                f"Sources:\n{sources_block}\n\n"
                "Return only plain text."
            )

            gen_start = time.perf_counter()
            resp = model.generate_content(prompt)
            gen_elapsed = (time.perf_counter() - gen_start) * 1000
            
            text = getattr(resp, "text", None)
            if not text and hasattr(resp, "candidates") and resp.candidates:
                try:
                    parts = []
                    for c in resp.candidates:
                        if hasattr(c, "content") and getattr(c.content, "parts", None):
                            for p in c.content.parts:
                                if hasattr(p, "text") and isinstance(p.text, str):
                                    parts.append(p.text)
                    text = "\n".join(parts)
                except Exception:
                    text = None
            
            total_elapsed = (time.perf_counter() - t_start) * 1000
            if text and text.strip():
                _log_timing("Gemini Standard Summary", total_elapsed, f"model={model_used} (generation: {gen_elapsed:.0f}ms)")
                return text.strip()
            else:
                _log_timing("Gemini Standard Summary", total_elapsed, "FAILED (empty response)")
                return None
        except Exception as e:
            total_elapsed = (time.perf_counter() - t_start) * 1000
            _log_timing("Gemini Standard Summary", total_elapsed, f"FAILED: {e}")
            return None

    def get_best_answer(self, query: str, *, hl: str = "en", gl: str = "us", num_results: int = 8) -> tuple[str, str]:
        """Return (text, source), where source is 'ai_overview' or 'gemini' or 'results' or 'none'."""
        ai = self.get_ai_overview(query, hl=hl, gl=gl, location=self.location)
        text = self.flatten_ai_overview(ai)
        if text:
            return text, "ai_overview"

        results = self.get_search_results(query, hl=hl, gl=gl, location=self.location, num_results=num_results)
        if not results:
            return "No results available for this query.", "none"

        summary = self._summarize_with_gemini(query, results)
        if summary:
            return summary, "gemini"

        formatted = self.format_results(results)
        return (formatted or "No results available for this query."), "results"

    def search_and_print(self, query: str) -> None:
        """Execute search and print only the final response with its source label."""
        text, source = self.get_best_answer(query)
        label = {
            "ai_overview": "[AI Overview]",
            "gemini": "[Gemini Summary]",
            "results": "[Search Results]",
            "none": "[No Result]",
        }.get(source, "")
        if label:
            print(f"{label} {text}")
        else:
            print(text)
