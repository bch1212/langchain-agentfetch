"""Thin HTTP client around the AgentFetch REST API.

Kept separate from the Tool subclasses so we can unit-test the client
independently and so the Tool layer stays declarative.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import httpx

DEFAULT_BASE_URL = "https://api.agentfetch.dev"


class AgentFetchClient:
    """Sync client for the AgentFetch REST API.

    For LangChain Tools we use the sync client; LangChain has both sync (.run)
    and async (.arun) entry points, but the sync path is simpler and the
    underlying httpx call releases the GIL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("AGENTFETCH_API_KEY") or os.getenv(
            "AGENTFETCH_KEY"
        )
        if not self.api_key:
            raise ValueError(
                "AgentFetch API key required. Pass api_key=... or set "
                "AGENTFETCH_API_KEY env var. Get a free key (500 fetches/mo) "
                "at https://www.agentfetch.dev/."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            headers={"X-AgentFetch-Key": self.api_key, "User-Agent": "langchain-agentfetch/0.1.0"},
        )

    def _post(self, path: str, json: dict) -> dict:
        resp = self._client.post(f"{self.base_url}{path}", json=json)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        resp = self._client.get(f"{self.base_url}{path}", params=params or {})
        resp.raise_for_status()
        return resp.json()

    def fetch(
        self,
        url: str,
        max_tokens: Optional[int] = None,
        format: str = "markdown",
        use_cache: bool = True,
    ) -> dict:
        return self._post(
            "/fetch",
            {
                "url": url,
                "max_tokens": max_tokens,
                "format": format,
                "use_cache": use_cache,
            },
        )

    def fetch_batch(
        self,
        urls: list[str],
        max_tokens_each: Optional[int] = None,
        use_cache: bool = True,
    ) -> dict:
        return self._post(
            "/fetch/batch",
            {"urls": urls, "max_tokens_each": max_tokens_each, "use_cache": use_cache},
        )

    def estimate(self, url: str) -> dict:
        return self._get("/estimate", {"url": url})

    def search(
        self, query: str, num_results: int = 3, max_tokens_each: int = 2000
    ) -> dict:
        return self._post(
            "/search",
            {
                "query": query,
                "num_results": num_results,
                "max_tokens_each": max_tokens_each,
            },
        )
