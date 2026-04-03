from langgraph.graph import StateGraph, START
from src.agent.state import State
from src.agent.nodes import call_model, call_tools, should_continue
from src.agent.tracing import get_langfuse_callback


builder = StateGraph(State)

builder.add_node("agent", call_model)
builder.add_node("tools", call_tools)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", "__end__"])
builder.add_edge("tools", "agent")

_compiled = builder.compile()

# Attach Langfuse callback at graph level so all invocations are traced,
# including calls from LangGraph Studio which don't use get_run_config().
_langfuse_handler = get_langfuse_callback()
if _langfuse_handler:
    graph = _compiled.with_config({"callbacks": [_langfuse_handler]})
else:
    graph = _compiled
