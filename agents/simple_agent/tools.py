from langchain.tools import tool
from tavily import TavilyClient
from typing import Dict, Any

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint


load_dotenv("../../.env")

@tool
def search_web(query: str) -> Dict[str, Any]:
    """
    Search the web for information
    """
    client = TavilyClient()

    return client.search(query)

# result = search_web.invoke("What is USD/EUR rate today?")

# print(result)

agent = create_agent(
    model="google_genai:gemini-3-flash-preview", 
    tools=[search_web])

question = HumanMessage("What is the maximum USD/EUR exchange rate in the current month?")

result = agent.invoke({"messages": [question]})

pprint(result)