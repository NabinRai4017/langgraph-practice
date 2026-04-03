from src.agent.graph import graph
from src.agent.tracing import get_run_config

config = get_run_config(session_id="chat-session", user_id="nabin")

print("Chat with your LangGraph agent. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit"):
        break
    if not user_input:
        continue

    result = graph.invoke(
        {"messages": [("human", user_input)]},
        config=config,
    )

    print(f"Agent: {result['messages'][-1].content}\n")
