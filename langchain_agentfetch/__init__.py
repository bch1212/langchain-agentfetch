"""LangChain Tool + Toolkit wrappers for AgentFetch.

Usage:

    from langchain_agentfetch import AgentFetchTool

    tool = AgentFetchTool(api_key="af_xxx")
    result = tool.run("https://news.ycombinator.com")

For an agent with all four AgentFetch tools:

    from langchain_agentfetch import AgentFetchToolkit
    from langchain.agents import AgentExecutor, create_tool_calling_agent

    toolkit = AgentFetchToolkit(api_key="af_xxx")
    agent = create_tool_calling_agent(llm, toolkit.get_tools(), prompt)
"""
from langchain_agentfetch.tool import (
    AgentFetchTool,
    EstimateTokensTool,
    FetchMultipleTool,
    SearchAndFetchTool,
)
from langchain_agentfetch.toolkit import AgentFetchToolkit

__all__ = [
    "AgentFetchTool",
    "EstimateTokensTool",
    "FetchMultipleTool",
    "SearchAndFetchTool",
    "AgentFetchToolkit",
]
__version__ = "0.1.0"
