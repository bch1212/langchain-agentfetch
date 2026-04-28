"""Toolkit that bundles all four AgentFetch tools as one drop-in for LangChain agents."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import BaseTool, BaseToolkit

from langchain_agentfetch.tool import (
    AgentFetchTool,
    EstimateTokensTool,
    FetchMultipleTool,
    SearchAndFetchTool,
)


class AgentFetchToolkit(BaseToolkit):
    """All four AgentFetch tools, configured with a shared API key.

    Example:

        from langchain_agentfetch import AgentFetchToolkit
        from langchain.agents import AgentExecutor, create_tool_calling_agent

        toolkit = AgentFetchToolkit(api_key="af_xxx")
        agent = create_tool_calling_agent(llm, toolkit.get_tools(), prompt)
    """

    api_key: Optional[str] = None
    base_url: str = "https://api.agentfetch.dev"

    def get_tools(self) -> List[BaseTool]:
        kwargs = {"api_key": self.api_key, "base_url": self.base_url}
        return [
            AgentFetchTool(**kwargs),
            EstimateTokensTool(**kwargs),
            FetchMultipleTool(**kwargs),
            SearchAndFetchTool(**kwargs),
        ]
