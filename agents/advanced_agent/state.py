from dotenv import load_dotenv
from pprint import pprint
from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv("../../.env", verbose=True)



class UserPreferences(AgentState):
    greetings: str


@tool
def read_greetings(runtime: ToolRuntime) -> str:
    """Read user greetings"""
    try:
        return runtime.context["greetings"]
    except KeyError:
        return "Hi!"

@tool
def update_greetings(greetings: str, runtime: ToolRuntime) -> Command:
    """Update preferred greetings"""
    return Command(update={
        "greetings": greetings,
        "messages": [ToolMessage("Successfully update user preferred greetings", tool_call_id=runtime.tool_call_id)]
    })

agent = create_agent(
    "google_genai:gemini-3-flash-preview",
    tools=[read_greetings, update_greetings],
    checkpointer=InMemorySaver(),
    state_schema=UserPreferences)

response = agent.invoke(
    {"messages": [HumanMessage("Hi! How are you? I'm Bob, but please always call me - Your majesty")]}, 
    {"configurable": {"thread_id": "1"}})

pprint(response)


response = agent.invoke(
    {"messages": [HumanMessage("I'm good thanks")]},
    {"configurable": {"thread_id": "1"}}
)

pprint(response)
