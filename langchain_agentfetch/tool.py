"""LangChain Tool subclasses wrapping AgentFetch endpoints.

Each Tool maps 1:1 to a hosted endpoint. The descriptions are tuned for
LangChain agents — the LLM uses them to decide when to call.
"""
from __future__ import annotations

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_agentfetch._client import AgentFetchClient


# --- fetch_url ---------------------------------------------------------------


class FetchUrlInput(BaseModel):
    url: str = Field(description="The URL to fetch.")
    max_tokens: Optional[int] = Field(
        default=None,
        description=(
            "Hard cap on response size in tokens. Pass this if you're tight on "
            "context window — cheaper than over-fetching."
        ),
    )


class AgentFetchTool(BaseTool):
    """Fetch any URL and return clean Markdown with token count + 6h caching.

    Use this instead of generic web fetch when you care about (a) blowing your
    context window, (b) cached re-reads of the same URL, (c) JS-rendered
    pages, or (d) PDFs. Auto-routes to the cheapest effective fetcher.
    """

    name: str = "agentfetch_fetch_url"
    description: str = (
        "Fetch any web URL and return clean, LLM-ready Markdown with token "
        "count and metadata. Auto-routes to the cheapest effective fetcher "
        "(Trafilatura → Jina → FireCrawl for JS pages → pypdf for PDFs). "
        "Pass max_tokens to cap response size. 6-hour cache built in. Use "
        "this whenever you need the contents of a specific URL."
    )
    args_schema: Type[BaseModel] = FetchUrlInput

    api_key: Optional[str] = None
    base_url: str = "https://api.agentfetch.dev"

    def _client(self) -> AgentFetchClient:
        return AgentFetchClient(api_key=self.api_key, base_url=self.base_url)

    def _run(
        self,
        url: str,
        max_tokens: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self._client().fetch(url=url, max_tokens=max_tokens)
        # Return JSON-shaped string so the LLM sees the metadata + markdown.
        # LangChain Tools are expected to return a string; structured output
        # is handled by the agent's parser.
        if not result.get("success"):
            return f"Fetch failed: {result.get('error', 'unknown error')}"
        return result.get("markdown", "")


# --- estimate_tokens ---------------------------------------------------------


class EstimateTokensInput(BaseModel):
    url: str = Field(description="The URL to estimate.")


class EstimateTokensTool(BaseTool):
    """Estimate token count of a URL WITHOUT fetching the body.

    Use BEFORE agentfetch_fetch_url when you're unsure whether a URL fits
    your remaining context budget. ~10× cheaper than a full fetch.
    """

    name: str = "agentfetch_estimate_tokens"
    description: str = (
        "Estimate the token count of a URL's content without performing a "
        "full fetch. Use this BEFORE fetch_url when you're unsure if a URL "
        "fits your context window. Returns estimated_tokens + a confident "
        "flag (false when the server omits Content-Length)."
    )
    args_schema: Type[BaseModel] = EstimateTokensInput

    api_key: Optional[str] = None
    base_url: str = "https://api.agentfetch.dev"

    def _client(self) -> AgentFetchClient:
        return AgentFetchClient(api_key=self.api_key, base_url=self.base_url)

    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self._client().estimate(url=url)
        est = result.get("estimated_tokens")
        confident = result.get("confident", False)
        if est is None:
            return (
                f"Could not estimate tokens for {url} (server did not return "
                f"Content-Length). Fetch with a max_tokens cap instead."
            )
        return f"Estimated {est:,} tokens (confident={confident}) for {url}."


# --- fetch_multiple ----------------------------------------------------------


class FetchMultipleInput(BaseModel):
    urls: list[str] = Field(description="1–20 URLs to fetch concurrently.")
    max_tokens_each: Optional[int] = Field(
        default=None,
        description="Per-result cap on response size. Total ≈ len(urls) * max_tokens_each.",
    )


class FetchMultipleTool(BaseTool):
    """Fetch up to 20 URLs concurrently. Returns a JSON array of results."""

    name: str = "agentfetch_fetch_multiple"
    description: str = (
        "Fetch up to 20 URLs in parallel. Each result is the same shape as "
        "fetch_url. Use when you have a list of links and want them retrieved "
        "concurrently rather than one at a time."
    )
    args_schema: Type[BaseModel] = FetchMultipleInput

    api_key: Optional[str] = None
    base_url: str = "https://api.agentfetch.dev"

    def _client(self) -> AgentFetchClient:
        return AgentFetchClient(api_key=self.api_key, base_url=self.base_url)

    def _run(
        self,
        urls: list[str],
        max_tokens_each: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self._client().fetch_batch(urls=urls, max_tokens_each=max_tokens_each)
        results = result.get("results", [])
        # Return a compact summary that an LLM can reason about + the markdown
        # of each successful fetch.
        out = []
        for r in results:
            if r.get("success"):
                out.append(
                    f"--- {r['url']} ({r['metadata']['token_count']} tokens) ---\n"
                    f"{r.get('markdown', '')}"
                )
            else:
                out.append(f"--- {r['url']} (FAILED: {r.get('error')}) ---")
        return "\n\n".join(out)


# --- search_and_fetch --------------------------------------------------------


class SearchAndFetchInput(BaseModel):
    query: str = Field(description="Search query string (2–500 chars).")
    num_results: int = Field(default=3, description="Top N results to fetch (1–10).")
    max_tokens_each: int = Field(default=2000, description="Per-result token cap.")


class SearchAndFetchTool(BaseTool):
    """Web search + fetch top N results in one round-trip.

    Use when you have a research question rather than a specific URL.
    """

    name: str = "agentfetch_search_and_fetch"
    description: str = (
        "Search the web and return clean content from the top N results. "
        "Use when you have a research question (e.g. 'find the latest docs "
        "on X library', 'recent news about Y') rather than specific URLs. "
        "Saves you the search-then-fetch dance."
    )
    args_schema: Type[BaseModel] = SearchAndFetchInput

    api_key: Optional[str] = None
    base_url: str = "https://api.agentfetch.dev"

    def _client(self) -> AgentFetchClient:
        return AgentFetchClient(api_key=self.api_key, base_url=self.base_url)

    def _run(
        self,
        query: str,
        num_results: int = 3,
        max_tokens_each: int = 2000,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self._client().search(
            query=query, num_results=num_results, max_tokens_each=max_tokens_each
        )
        if "error" in result:
            return f"Search failed: {result['error']}"
        out = [f"Query: {query}\n"]
        for r in result.get("results", []):
            if r.get("success"):
                out.append(
                    f"--- {r['url']} ({r['metadata']['token_count']} tokens) ---\n"
                    f"{r.get('markdown', '')}"
                )
        return "\n\n".join(out)
