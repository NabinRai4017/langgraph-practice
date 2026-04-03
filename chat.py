from langchain_core.messages import ToolMessage
from src.agent.graph import graph
from src.agent.tracing import get_run_config

# Each chat session gets a unique thread_id so the checkpointer
# can persist and resume state across the interrupt/approval cycle.
config = get_run_config()

print("Chat with your LangGraph agent (Human-in-the-Loop enabled).")
print("You will be asked to approve tool calls before they execute.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit"):
        break
    if not user_input:
        continue

    graph.invoke({"messages": [("human", user_input)]}, config=config)

    # After invoking, check if the graph is paused before a tool call.
    while graph.get_state(config).next == ("tools",):
        state = graph.get_state(config)
        pending = state.values["messages"][-1].tool_calls
        print("\n[Agent wants to use tools:]")
        for tc in pending:
            print(f"  {tc['name']} => {tc['args']}")

        approval = input("\nApprove tool use? (y/n): ").strip().lower()

        if approval == "y":
            # Resume — let the tools execute and agent continue.
            graph.invoke(None, config=config)
        else:
            # Rejection: inject a ToolMessage for each pending call telling
            # the agent the tool was rejected, then resume so the agent
            # generates a fresh answer without executing any real tools.
            rejections = [
                ToolMessage(
                    content="Tool use rejected by user. Answer from your own knowledge.",
                    tool_call_id=tc["id"],
                )
                for tc in pending
            ]
            # as_node="tools" tells LangGraph these messages already came
            # from the tools node, so the interrupt won't fire again.
            graph.update_state(config, {"messages": rejections}, as_node="tools")
            graph.invoke(None, config=config)

    final = graph.get_state(config).values["messages"][-1]
    print(f"\nAgent: {final.content}\n")
