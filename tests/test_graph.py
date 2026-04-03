import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph import graph


@pytest.fixture
def mock_model():
    with patch("src.agent.nodes.model") as m:
        yield m


def test_graph_compiles():
    assert graph is not None


def test_simple_invoke(mock_model):
    mock_model.invoke.return_value = AIMessage(content="Hello!")
    result = graph.invoke({"messages": [HumanMessage(content="Hi")]})
    assert result["messages"][-1].content == "Hello!"
