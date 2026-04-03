import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph import graph, checkpointer


def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@pytest.fixture
def mock_model():
    with patch("src.agent.nodes.model") as m:
        yield m


def test_graph_has_checkpointer():
    assert checkpointer is not None


def test_graph_pauses_before_tools(mock_model):
    """Graph should interrupt before tools when agent makes a tool call."""
    tool_call = {
        "id": "call_abc",
        "name": "tavily_search",
        "args": {"query": "LangGraph HITL"},
        "type": "tool_call",
    }
    mock_model.invoke.return_value = AIMessage(content="", tool_calls=[tool_call])

    config = make_config("test-pause")
    graph.invoke({"messages": [HumanMessage(content="Search LangGraph")]}, config=config)

    state = graph.get_state(config)
    assert state.next == ("tools",), "Graph should be paused before tools"


def test_graph_resumes_after_approval(mock_model):
    """Resuming with None should execute tools and return final answer."""
    tool_call = {
        "id": "call_xyz",
        "name": "tavily_search",
        "args": {"query": "capital of France"},
        "type": "tool_call",
    }
    mock_model.invoke.side_effect = [
        AIMessage(content="", tool_calls=[tool_call]),
        AIMessage(content="Paris."),
    ]

    config = make_config("test-resume")

    with patch("src.agent.nodes.tools") as mock_tools:
        mock_tools[0].name = "tavily_search"
        mock_tools[0].invoke.return_value = "Paris is the capital of France."
        mock_tools.__iter__ = lambda self: iter([mock_tools[0]])

        graph.invoke({"messages": [HumanMessage(content="Capital of France?")]}, config=config)
        assert graph.get_state(config).next == ("tools",)

        graph.invoke(None, config=config)

    state = graph.get_state(config)
    assert state.next == ()
    assert state.values["messages"][-1].content == "Paris."


def test_graph_handles_rejection(mock_model):
    """Rejecting tool use injects ToolMessage rejections so the agent answers without real tools."""
    from langchain_core.messages import ToolMessage
    tool_call = {
        "id": "call_rej",
        "name": "tavily_search",
        "args": {"query": "something"},
        "type": "tool_call",
    }
    mock_model.invoke.side_effect = [
        AIMessage(content="", tool_calls=[tool_call]),
        AIMessage(content="I'll answer without tools."),
    ]

    config = make_config("test-reject")
    graph.invoke({"messages": [HumanMessage(content="Tell me something")]}, config=config)
    assert graph.get_state(config).next == ("tools",)

    # Inject rejection ToolMessages, then resume so agent re-runs without real tools.
    state = graph.get_state(config)
    pending = state.values["messages"][-1].tool_calls
    rejections = [
        ToolMessage(
            content="Tool use rejected by user. Answer from your own knowledge.",
            tool_call_id=tc["id"],
        )
        for tc in pending
    ]
    graph.update_state(config, {"messages": rejections}, as_node="tools")
    graph.invoke(None, config=config)

    state = graph.get_state(config)
    assert state.values["messages"][-1].content == "I'll answer without tools."
