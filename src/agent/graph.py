from langgraph.graph import StateGraph, START
from src.agent.state import State
from src.agent.nodes import call_model, call_tools, should_continue


builder = StateGraph(State)

builder.add_node("agent", call_model)
builder.add_node("tools", call_tools)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", "__end__"])
builder.add_edge("tools", "agent")

graph = builder.compile()
