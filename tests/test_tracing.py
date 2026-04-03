import pytest
from unittest.mock import patch


def test_get_run_config_returns_thread_id_without_langfuse_keys():
    with patch.dict("os.environ", {}, clear=True):
        with patch("dotenv.load_dotenv"):
            from importlib import reload
            import src.agent.tracing as tracing
            reload(tracing)
            config = tracing.get_run_config()
            assert "configurable" in config
            assert "thread_id" in config["configurable"]
            assert "callbacks" not in config


def test_get_run_config_returns_callbacks_with_keys():
    env = {
        "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
        "LANGFUSE_SECRET_KEY": "sk-lf-test",
        "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
    }
    with patch.dict("os.environ", env):
        with patch("langfuse.langchain.CallbackHandler") as mock_handler:
            mock_handler.return_value = object()
            from importlib import reload
            import src.agent.tracing as tracing
            reload(tracing)
            config = tracing.get_run_config(session_id="s1", user_id="u1")
            assert "callbacks" in config
            assert len(config["callbacks"]) == 1
