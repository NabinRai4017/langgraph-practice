"""
Integration test — hits the real DeepSeek API and sends a trace to Langfuse.
Skipped automatically if LANGFUSE_PUBLIC_KEY or DEEPSEEK_API_KEY is not set.

Run explicitly with:
    pytest tests/test_langfuse_integration.py -v -s
"""
import os
import pytest
from src.agent.graph import graph
from src.agent.tracing import get_run_config


@pytest.mark.skipif(
    not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("DEEPSEEK_API_KEY"),
    reason="LANGFUSE_PUBLIC_KEY or DEEPSEEK_API_KEY not set",
)
def test_langfuse_trace_sent():
    config = get_run_config(thread_id="test-session-1", session_id="test-session-1", user_id="nabin")
    result = graph.invoke(
        {"messages": [("human", "What is the capital of France?")]},
        config=config,
    )
    last_message = result["messages"][-1]
    assert last_message.content, "Expected a non-empty response from the agent"
    print("\nResponse:", last_message.content)
    print("Check your Langfuse dashboard at https://cloud.langfuse.com")
