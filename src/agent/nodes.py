from langchain_core.messages import AIMessage
from src.agent.state import State
from src.agent.tools import tools
from src.agent.model import model


def call_model(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def call_tools(state: State) -> dict:
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_map = {t.name: t for t in tools}
        tool = tool_map[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(
            {"role": "tool", "content": str(result), "tool_call_id": tool_call["id"]}
        )
    return {"messages": results}


def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"
