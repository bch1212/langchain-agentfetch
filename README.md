# langchain-agentfetch

> Drop-in [LangChain](https://python.langchain.com/) Tool wrappers for [AgentFetch](https://www.agentfetch.dev) — token-budgeted web fetch for AI agents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/langchain-agentfetch)](https://pypi.org/project/langchain-agentfetch/)

Stop building your own web fetch / token-truncation / cache layer for every LangChain agent. AgentFetch handles routing (Trafilatura → Jina → FireCrawl → PDF), caching (6h Redis), and token budgeting in one tool.

## Install

```bash
pip install langchain-agentfetch
```

## Quick start

Get a free API key (500 fetches/mo, no credit card) at [agentfetch.dev](https://www.agentfetch.dev).

```python
import os
from langchain_agentfetch import AgentFetchTool

os.environ["AGENTFETCH_API_KEY"] = "af_xxx"

tool = AgentFetchTool()
result = tool.run({"url": "https://news.ycombinator.com", "max_tokens": 2000})
print(result)
```

## All four tools

| Tool | When to use |
|---|---|
| `AgentFetchTool` (`fetch_url`) | You have a specific URL to fetch |
| `EstimateTokensTool` | You want to know if a URL fits your context window before fetching |
| `FetchMultipleTool` | You have multiple URLs (search results, link lists) |
| `SearchAndFetchTool` | You have a research question, not specific URLs |

## Use the toolkit (recommended for agents)

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_agentfetch import AgentFetchToolkit

toolkit = AgentFetchToolkit(api_key="af_xxx")
tools = toolkit.get_tools()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Use AgentFetch tools to read web pages "
               "and answer questions. Always estimate tokens before fetching long URLs."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

executor.invoke({"input": "What's on the Hacker News front page right now?"})
```

The agent will:
1. Use `agentfetch_search_and_fetch` or `agentfetch_fetch_url` to get the page
2. Get clean Markdown back, already token-truncated
3. Reason over it without blowing the context window

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `AGENTFETCH_API_KEY` | required | Get one free at agentfetch.dev |
| `AGENTFETCH_BASE_URL` | `https://api.agentfetch.dev` | Override only if self-hosting |

You can also pass `api_key=` and `base_url=` directly to any Tool / Toolkit constructor.

## Why a token-aware fetch tool

LangChain's default tools (like `requests_get`) return raw HTML/JSON with no truncation, no awareness of your context budget, and no caching. That works at toy scale; it falls over the moment your agent fetches a 50KB news article into a 4K context window.

AgentFetch was built for that case:
- **Pre-fetch token estimation** so the agent can skip a URL it can't afford
- **Server-side `max_tokens` truncation** before the response leaves the API
- **6-hour cache** for repeat fetches (~$0.0001 each)
- **Auto-routing** so the agent doesn't pick between Jina, FireCrawl, and pypdf manually

## Pricing

Same pay-per-call pricing as the AgentFetch API: from $0.001/fetch, 500 free on signup. No subscription required. See [agentfetch.dev/pricing](https://www.agentfetch.dev/pricing).

## License

MIT.
