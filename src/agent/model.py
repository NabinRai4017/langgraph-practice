import os
from dotenv import load_dotenv

load_dotenv()

_provider = os.getenv("LLM_PROVIDER", "deepseek")

if _provider == "openai":
    from langchain_openai import ChatOpenAI
    from src.agent.tools import tools
    model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)
elif _provider == "anthropic":
    from langchain_anthropic import ChatAnthropic
    from src.agent.tools import tools
    model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).bind_tools(tools)
else:  # deepseek (default)
    from langchain_deepseek import ChatDeepSeek
    from src.agent.tools import tools
    model = ChatDeepSeek(model="deepseek-chat", temperature=0).bind_tools(tools)
