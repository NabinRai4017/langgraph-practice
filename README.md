# LangGraph Agent

A minimal LangGraph agent project wired to run with DeepSeek (default), OpenAI, or Anthropic.

## Releases

### [v1.1.0](https://github.com/NabinRai4017/langgraph-practice/releases/tag/v1.1.0) — Tavily Search Integration
- Replaced placeholder search tool with **Tavily** (`TavilySearchResults`, `max_results=3`)
- Added `langchain-tavily` dependency
- Added `TAVILY_API_KEY` to `.env.example`
- All 13 pytest tests passing

```bash
git checkout v1.1.0
```

### [v1](https://github.com/NabinRai4017/langgraph-practice/releases/tag/v1) — Sample Tests
Initial working version of the LangGraph agent. Includes:
- LangGraph agent with DeepSeek as the default LLM (`deepseek-chat`)
- Switchable LLM providers: DeepSeek, OpenAI, Anthropic
- Langfuse tracing (v4) auto-attached at graph level — works in LangGraph Studio and `chat.py`
- 13 pytest tests covering graph, state, tools, tracing, and Langfuse integration
- Interactive `chat.py` script for testing messages locally

```bash
git checkout v1
```

## Project Structure

```
langgraph/
├── pyproject.toml          # Dependencies and project config
├── langgraph.json          # LangGraph server config
├── .env.example            # Environment variable template
├── chat.py                 # Interactive CLI for testing messages
├── src/
│   └── agent/
│       ├── state.py        # Graph state definition
│       ├── model.py        # LLM setup (provider switching)
│       ├── tools.py        # Tool definitions
│       ├── nodes.py        # Node functions and routing logic
│       ├── tracing.py      # Langfuse tracing setup
│       └── graph.py        # Graph assembly, compile, and tracing attachment
└── tests/
    ├── test_graph.py                 # Graph unit tests
    ├── test_state.py                 # State tests
    ├── test_tools.py                 # Tool tests
    ├── test_tracing.py               # Tracing unit tests
    └── test_langfuse_integration.py  # Live Langfuse integration test
```

## Graph Flow

```
START → agent → tools → agent → ... → END
```

The agent calls tools when needed and loops back until it has a final answer.

## Setup

**1. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

**2. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and add your DeepSeek API key:

```
DEEPSEEK_API_KEY=your_key_here
```

Get your key at [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys).

**3. Run the dev server**

```bash
langgraph dev
```

This starts a local LangGraph Studio server at `http://localhost:8123` where you can interact with and visualize the graph.

## Switching LLM Providers

Set `LLM_PROVIDER` in your `.env` file:

| Value | Model used | Key needed |
|-------|-----------|------------|
| `deepseek` (default) | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |

## Adding Tools

Edit `src/agent/tools.py` and add new `@tool` functions, then include them in the `tools` list:

```python
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """Describe what this tool does."""
    return "result"

tools = [my_tool]
```

## Extending State

Edit `src/agent/state.py` to add custom state fields:

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    user_id: str
    context: dict
```

## Tracing with Langfuse

This project uses [Langfuse](https://langfuse.com) for observability instead of LangSmith.

**1. Add your Langfuse keys to `.env`:**

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Get them from [cloud.langfuse.com](https://cloud.langfuse.com) → Project Settings → API Keys.

**2. Pass the config when invoking the graph:**

```python
from src.agent.graph import graph
from src.agent.tracing import get_run_config

config = get_run_config(session_id="session-1", user_id="user-abc")
result = graph.invoke({"messages": [("human", "Hello!")]}, config=config)
```

If `LANGFUSE_PUBLIC_KEY` is not set, `get_run_config()` returns `{}` and tracing is silently skipped.

**Self-hosting Langfuse:** set `LANGFUSE_HOST=http://localhost:3000` in `.env`.

## Running Tests

```bash
pytest
```
