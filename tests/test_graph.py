import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agent.graph import graph


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
    result = graph.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
    assert result["messages"][-1].content == "Paris is the capital of France."


def test_agent_called_once_without_tools(mock_model):
    mock_model.invoke.return_value = AIMessage(content="Done.")
    graph.invoke({"messages": [HumanMessage(content="Hello")]})
    assert mock_model.invoke.call_count == 1


def test_tool_call_loops_back_to_agent(mock_model):
    """Agent requests a tool call, tool runs, then agent responds with final answer."""
    tool_call = {
        "id": "call_123",
        "name": "search",
        "args": {"query": "capital of France"},
        "type": "tool_call",
    }
    mock_model.invoke.side_effect = [
        AIMessage(content="", tool_calls=[tool_call]),
        AIMessage(content="Paris."),
    ]

    with patch("src.agent.nodes.tools", [MagicMock(name="search", invoke=lambda args: "Paris")]) as mock_tools:
        mock_tools[0].name = "search"
        mock_tools[0].invoke.return_value = "Paris"
        result = graph.invoke({"messages": [HumanMessage(content="Capital of France?")]})

    assert mock_model.invoke.call_count == 2
    assert result["messages"][-1].content == "Paris."
