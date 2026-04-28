"""Smoke tests — no network. Verify imports and Tool metadata are sane."""
from __future__ import annotations


def test_imports():
    from langchain_agentfetch import (
        AgentFetchTool,
        EstimateTokensTool,
        FetchMultipleTool,
        SearchAndFetchTool,
        AgentFetchToolkit,
    )

    assert AgentFetchTool.__name__ == "AgentFetchTool"
    assert AgentFetchToolkit.__name__ == "AgentFetchToolkit"


def test_tool_metadata():
    from langchain_agentfetch import AgentFetchTool

    t = AgentFetchTool(api_key="af_test")
    assert t.name == "agentfetch_fetch_url"
    # The description must mention the agent-relevant capabilities so the LLM
    # picks this tool when appropriate.
    desc = t.description.lower()
    assert "max_tokens" in desc
    assert "cache" in desc
    assert "markdown" in desc


def test_toolkit_returns_all_four():
    from langchain_agentfetch import AgentFetchToolkit

    tools = AgentFetchToolkit(api_key="af_test").get_tools()
    assert len(tools) == 4
    assert {t.name for t in tools} == {
        "agentfetch_fetch_url",
        "agentfetch_estimate_tokens",
        "agentfetch_fetch_multiple",
        "agentfetch_search_and_fetch",
    }


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("AGENTFETCH_API_KEY", raising=False)
    monkeypatch.delenv("AGENTFETCH_KEY", raising=False)
    from langchain_agentfetch._client import AgentFetchClient

    try:
        AgentFetchClient()
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "API key required" in str(e)
