from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

search = TavilySearch(max_results=3)

tools = [search]
