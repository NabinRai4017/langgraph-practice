from langchain_core.tools import tool


@tool
def search(query: str) -> str:
    """Search for information on the web. Replace this with a real search tool."""
    return f"Results for: {query}"


tools = [search]
