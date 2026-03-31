"""
Optional web search service for up-to-date IT context enrichment.
"""

from __future__ import annotations

import html
import logging
import os
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, unquote

import aiohttp


logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    provider: str
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_context_block(self) -> str:
        snippet = self.snippet.strip() if self.snippet else "No snippet available."
        return f"[{self.rank}] {self.title}\nURL: {self.url}\nSnippet: {snippet}"


class _DuckDuckGoHTMLParser(HTMLParser):
    """Minimal parser for DuckDuckGo HTML search pages."""

    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._current: Optional[Dict[str, str]] = None
        self._capture_title = False
        self._capture_snippet = False

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        class_name = attr_map.get("class", "")

        if tag == "a" and ("result__a" in class_name or "result-link" in class_name):
            href = attr_map.get("href", "").strip()
            if href:
                self._current = {
                    "title": "",
                    "url": _normalize_duckduckgo_href(href),
                    "snippet": "",
                }
                self._capture_title = True
                self.results.append(self._current)
            return

        if self._current and ("result__snippet" in class_name or "result-snippet" in class_name):
            self._capture_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._capture_title = False
        if tag in {"a", "div", "span"}:
            self._capture_snippet = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or not self._current:
            return

        if self._capture_title:
            current = self._current["title"]
            self._current["title"] = f"{current} {text}".strip()
            return

        if self._capture_snippet:
            current = self._current["snippet"]
            self._current["snippet"] = f"{current} {text}".strip()


def _normalize_duckduckgo_href(href: str) -> str:
    if href.startswith("//"):
        href = f"https:{href}"
    if href.startswith("/"):
        href = urljoin("https://duckduckgo.com", href)

    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc:
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return href


class WebSearchService:
    def __init__(self) -> None:
        self.enabled = os.getenv("VALLM_ENABLE_WEB_SEARCH", "true").lower() == "true"
        self.provider = os.getenv("VALLM_WEB_SEARCH_PROVIDER", "auto").strip().lower()
        self.max_results = max(1, int(os.getenv("VALLM_WEB_SEARCH_MAX_RESULTS", "5")))
        self.timeout_seconds = float(os.getenv("VALLM_WEB_SEARCH_TIMEOUT_SECONDS", "20"))
        self.region = os.getenv("VALLM_WEB_SEARCH_REGION", "wt-wt")
        self.safe_search = os.getenv("VALLM_WEB_SEARCH_SAFE", "moderate")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        self.user_agent = os.getenv(
            "VALLM_WEB_SEARCH_USER_AGENT",
            "VaLLM-IT-Specialist/1.0 (+https://joelwembo.com)",
        )

    def is_available(self) -> bool:
        return self.enabled

    async def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        query = (query or "").strip()
        if not query:
            return []

        limit = max_results or self.max_results
        provider = self._select_provider()
        providers_to_try = [provider]
        if provider != "duckduckgo":
            providers_to_try.append("duckduckgo")

        for candidate in providers_to_try:
            try:
                results = await self._search_with_provider(candidate, query, limit)
                if results:
                    return [result.to_dict() for result in results[:limit]]
            except Exception as exc:
                logger.warning("Web search provider %s failed: %s", candidate, exc)

        return []

    def format_results_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No live web results available."

        blocks: List[str] = []
        for idx, item in enumerate(results, start=1):
            result = WebSearchResult(
                title=str(item.get("title", "")).strip() or f"Result {idx}",
                url=str(item.get("url", "")).strip(),
                snippet=str(item.get("snippet", "")).strip(),
                provider=str(item.get("provider", "web")),
                rank=int(item.get("rank", idx)),
            )
            blocks.append(result.to_context_block())
        return "\n\n".join(blocks)

    def _select_provider(self) -> str:
        if self.provider and self.provider != "auto":
            return self.provider
        if self.tavily_api_key:
            return "tavily"
        if self.serpapi_api_key:
            return "serpapi"
        return "duckduckgo"

    async def _search_with_provider(self, provider: str, query: str, limit: int) -> List[WebSearchResult]:
        if provider == "tavily":
            return await self._search_tavily(query, limit)
        if provider == "serpapi":
            return await self._search_serpapi(query, limit)
        return await self._search_duckduckgo(query, limit)

    async def _search_tavily(self, query: str, limit: int) -> List[WebSearchResult]:
        if not self.tavily_api_key:
            return []

        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": limit,
            "topic": "general",
        }
        data = await self._request_json(
            "POST",
            "https://api.tavily.com/search",
            json=payload,
        )
        items = data.get("results") or []
        return [
            WebSearchResult(
                title=str(item.get("title", "")).strip() or f"Result {idx}",
                url=str(item.get("url", "")).strip(),
                snippet=str(item.get("content", "")).strip(),
                provider="tavily",
                rank=idx,
            )
            for idx, item in enumerate(items[:limit], start=1)
            if item.get("url")
        ]

    async def _search_serpapi(self, query: str, limit: int) -> List[WebSearchResult]:
        if not self.serpapi_api_key:
            return []

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_api_key,
            "num": limit,
        }
        data = await self._request_json(
            "GET",
            f"https://serpapi.com/search.json?{urlencode(params)}",
        )
        items = data.get("organic_results") or []
        return [
            WebSearchResult(
                title=str(item.get("title", "")).strip() or f"Result {idx}",
                url=str(item.get("link", "")).strip(),
                snippet=str(item.get("snippet", "")).strip(),
                provider="serpapi",
                rank=idx,
            )
            for idx, item in enumerate(items[:limit], start=1)
            if item.get("link")
        ]

    async def _search_duckduckgo(self, query: str, limit: int) -> List[WebSearchResult]:
        params = {
            "q": query,
            "kl": self.region,
            "kp": self.safe_search,
        }
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml",
        }
        html_text = await self._request_text(
            "GET",
            f"https://html.duckduckgo.com/html/?{urlencode(params)}",
            headers=headers,
        )
        parser = _DuckDuckGoHTMLParser()
        parser.feed(html_text)

        results: List[WebSearchResult] = []
        for idx, item in enumerate(parser.results, start=1):
            title = html.unescape(item.get("title", "")).strip()
            url = item.get("url", "").strip()
            snippet = html.unescape(item.get("snippet", "")).strip()
            if not title or not url:
                continue
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    provider="duckduckgo",
                    rank=idx,
                )
            )
            if len(results) >= limit:
                break
        return results

    async def _request_json(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()

    async def _request_text(self, method: str, url: str, **kwargs: Any) -> str:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.text()


web_search_service = WebSearchService()
