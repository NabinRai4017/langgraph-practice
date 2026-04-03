import os
from dotenv import load_dotenv

load_dotenv()


def get_langfuse_callback():
    """Return a Langfuse CallbackHandler if credentials are configured, else None.

    In Langfuse v4, credentials are read automatically from env vars:
      LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
    """
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None

    from langfuse.langchain import CallbackHandler
    return CallbackHandler()


def get_run_config(
    thread_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Return a LangGraph-compatible config dict with thread_id and Langfuse tracing."""
    import uuid
    config: dict = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    handler = get_langfuse_callback()
    if handler:
        config["callbacks"] = [handler]

    return config
