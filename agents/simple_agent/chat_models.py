from dotenv import load_dotenv
from pprint import pprint
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from pydantic import BaseModel


load_dotenv("../../.env")

# if do not specify provider, could try to access it from Google Cloud/Vertex AI instead of AI Studio
model = init_chat_model(
    model="gemini-3-flash-preview", 
    model_provider="google_genai",
    temperature=0)

# response = model.invoke("What's the capital of the Moon?")
# pprint(response)


agent = create_agent(model=model)

# response = agent.invoke({
#     "messages": [HumanMessage("What's the capital of the Moon?")]
# })

# pprint(response)

# response = agent.invoke(
#     {"messages": [HumanMessage(content="What's the capital of the Moon?"),
#     AIMessage(content="The capital of the Moon is Luna City."),
#     HumanMessage(content="Interesting, tell me more about Luna City")]}
# )

# pprint(response)


# for token, metadata in agent.stream(
#     {"messages": [HumanMessage("What is AI Agent?")]},
#     stream_mode="messages"
# ):
#     if token.content:
#         print(token.content, end="", flush=True)

class ResponseTemplate(BaseModel):
    name: str
    age: int
    occupacy: str
    hobby: str


question = HumanMessage(content="Who is main character?")

# detective_agent = create_agent(
#     model=model,
#     system_prompt="You are detective writer. Create a user at the users request",
#     response_format=ResponseTemplate)

# response = detective_agent.invoke({"messages": [question]})

# pprint(response)

scifi_agent = create_agent(
    model=model,
    system_prompt="You are fiction writer. Create a user at the users request",
    response_format=ResponseTemplate)

response = scifi_agent.invoke({"messages": [question]})

pprint(response)
