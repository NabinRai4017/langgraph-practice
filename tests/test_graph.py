import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agent.graph import graph


def cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@pytest.fixture
def mock_model():
    with patch("src.agent.nodes.model") as m:
        yield m


def test_graph_compiles():
    assert graph is not None


def test_graph_has_expected_nodes():
    assert "agent" in graph.nodes
    assert "tools" in graph.nodes


def test_simple_invoke(mock_model):
    mock_model.invoke.return_value = AIMessage(content="Paris is the capital of France.")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]},
        config=cfg("t-simple"),
    )
    assert result["messages"][-1].content == "Paris is the capital of France."


def test_agent_called_once_without_tools(mock_model):
    mock_model.invoke.return_value = AIMessage(content="Done.")
    graph.invoke({"messages": [HumanMessage(content="Hello")]}, config=cfg("t-once"))
    assert mock_model.invoke.call_count == 1


def test_tool_call_loops_back_to_agent(mock_model):
    """Agent requests a tool call, tool runs, then agent responds with final answer."""
    tool_call = {
        "id": "call_123",
        "name": "tavily_search",
        "args": {"query": "capital of France"},
        "type": "tool_call",
    }
    mock_model.invoke.side_effect = [
        AIMessage(content="", tool_calls=[tool_call]),
        AIMessage(content="Paris."),
    ]

    config = cfg("t-loop")
    with patch("src.agent.nodes.tools", [MagicMock(name="tavily_search", invoke=lambda args: "Paris")]) as mock_tools:
        mock_tools[0].name = "tavily_search"
        mock_tools[0].invoke.return_value = "Paris"
        # First invoke — pauses before tools (HITL interrupt)
        graph.invoke({"messages": [HumanMessage(content="Capital of France?")]}, config=config)
        # Resume — approve tool use
        graph.invoke(None, config=config)

    assert mock_model.invoke.call_count == 2
    assert graph.get_state(config).values["messages"][-1].content == "Paris."
