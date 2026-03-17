from tavily import TavilyClient
from config import TAVILY_API_KEY

tavily = TavilyClient(api_key=TAVILY_API_KEY)

def search_web(query):

    response = tavily.search(query=query, max_results=2)

    results = ""

    for r in response["results"]:
        results += r["content"] + "\n"

    return results