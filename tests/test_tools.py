from src.agent.tools import search, tools


def test_tools_list_not_empty():
    assert len(tools) > 0


def test_tools_have_names():
    for tool in tools:
        assert tool.name is not None


def test_search_tool_exists():
    assert search is not None
    assert search.name == "tavily_search"
