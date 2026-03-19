from dotenv import load_dotenv

from pprint import pprint

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

load_dotenv("../../.env")

agent = create_agent(
    model="google_genai:gemini-3-flash-preview",
    checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}

question = HumanMessage(content="Hi, I'm Alex. My favourite subject is Math. Also I like to play footbal. What about you?")

response = agent.invoke(
    {"messages": [question]},
    config,)

pprint(response)

question = HumanMessage(content="Cool. Do you know my name?")

response = agent.invoke(
    {"messages": [question]},
    config,)

pprint(response)