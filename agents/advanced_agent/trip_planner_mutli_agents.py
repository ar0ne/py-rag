import asyncio
from dotenv import load_dotenv
from pprint import pprint
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import tool, ToolRuntime
from tavily import TavilyClient
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import ToolMessage, HumanMessage
from langgraph.types import Command
from typing import Any, Dict
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent



AI_MODEL = "google_genai:gemini-3-flash-preview"

load_dotenv("../../.env")


class TripPlanState(AgentState):
    origin: str
    destination: str
    hobby: str
    departure_date: str


RETRYABLE_MCP_CODES = {-32603}

class RetryMCPInterceptor:
    """Intercept MCP tool calls: retry transient failures, surface all errors gracefully.

    - Retryable McpError codes (e.g. -32603): retry with exponential backoff.
    - Non-retryable McpError codes (e.g. -32602): return error message immediately.
    - Any other exception (fetch failed, network errors, etc.): retry then return error message.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(code {exc.error.code}, attempt {attempt+1}/{self.max_retries}): {exc}")
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(attempt {attempt+1}/{self.max_retries}): {exc}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        print(f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )


async def main():
    ########################
    # Flight tickets (MCP)
    ########################
    travel_server_mcp_client = MultiServerMCPClient(
        {
            "travel_server": {
                    "transport": "streamable_http",
                    "url": "https://mcp.kiwi.com"
                }
        },
        tool_interceptors=[RetryMCPInterceptor()],
    )

    tools = await travel_server_mcp_client.get_tools()


    flight_agent = create_agent(
        model=AI_MODEL,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt="""
        You are travel agent. Search for flights to the desired destination.
        You are not allowed follow up questions, you must find the best flights based on the following criteria:
        - Price (the lowest, economy class)
        - Duration (the shortest)
        - Luggage (included)
        Look only for one-way tickets.
        You may need to make multiple searches to iteratively find the best options.
        You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
        If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
        Once you have found the best options, let the user know your shortlist of options.
        """
    )

    #########################
    # best places to visit
    #########################
    tavily_client = TavilyClient()

    @tool
    def web_search(query: str) -> Dict[str, Any]:
        """Search the web for the users query"""
        return tavily_client.search(query)


    suggest_agent = create_agent(
        model=AI_MODEL,
        tools=[web_search],
        system_prompt="""
        You are trip advisor. Search the best places to visit in destion location.
        No follow up questions. You must find the best places based on following criteria:
        - User's hobby
        - Price (the lowest or absent)
        - Uniqueness (architecture)
        You may need to make multiple searches to iteratively find the best options. 
        You have a suggested limit of 12 web searches. Count every web_search call you make.
        After 12 searches, you should stop searching and summarize the best options you have
        found so far.
        Provide only names separated by comma.""",
        checkpointer=InMemorySaver()
    )

    # book accomodations ?


    #########################
    # Coordinator
    #########################
    @tool
    async def search_flights(runtime: ToolRuntime) -> str:
        """Search flight tickets to the desired destination"""
        origin = runtime.state["origin"]
        destination = runtime.state["destination"]
        departure_date = runtime.state["departure_date"]
        response = await flight_agent.ainvoke(
            {"messages": [HumanMessage(content=f"Find flights from {origin} to {destination} on {departure_date}")]}
        )
        return response['messages'][-1].content


    @tool
    async def search_places_to_visit(runtime: ToolRuntime) -> str:
        """Search top places to visit in the desired destination"""
        destination = runtime.state["destination"]
        hobby = runtime.state["hobby"]
        response = await suggest_agent.ainvoke(
            {"messages": [HumanMessage(content=f"Find top 10 of the best places to visit in {destination} and where user could do: {hobby}")]}
        )
        return response['messages'][-1].content


    @tool
    def update_state(origin: str, destination: str, hobby: str, departure_date: str, runtime: ToolRuntime):
        """Update the state when you know all of the values: origin, destination, hobby.
        This tool must be called alone, without any other tool calls. It must complete and return to make,
        the information available to other tools."""
        return Command(update={
            "origin": origin,
            "destination": destination,
            "hobby": hobby,
            "departure_date": departure_date,
            "messages": [ToolMessage("Successfully updated the state", tool_call_id=runtime.tool_call_id)]
        })

    coordinator = create_agent(
        model=AI_MODEL,
        tools=[search_flights, search_places_to_visit, update_state],
        state_schema=TripPlanState,
        system_prompt="""
        You are travel agent coordinator.
        First find all the information you need to update the state. When you have the information, update the state.
        Once that has completed and returned, you can delegate the tasks 
        to your specialists for flights, places.
        Once you have received their answers, coordinate the perfect trip for me.
        """
    )

    response = await coordinator.ainvoke(
        {"messages": [HumanMessage("""I plan to visit Sri Lanka in February 2027. I plan to surf and visit local attractions.
        I'll need to buy flight tickets from Berlin. Any suggestions?""")]},
        {"configurable": {"thread_id": "1"}})

    pprint(response)


if __name__ == "__main__":
    asyncio.run(main())


