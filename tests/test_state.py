from langchain_core.messages import HumanMessage, AIMessage
from src.agent.state import State


def test_state_accepts_messages():
    state = State(messages=[HumanMessage(content="Hi")])
    assert len(state["messages"]) == 1


def test_state_message_types():
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ]
    state = State(messages=messages)
    assert state["messages"][0].content == "Hello"
    assert state["messages"][1].content == "Hi there!"
