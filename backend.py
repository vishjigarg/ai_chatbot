from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import os
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

load_dotenv()

class ChatState(BaseModel):
    messages: Annotated[List[BaseMessage], Field(description="List of chat messages in the conversation history."), add_messages]



hf_llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    task="conversational",
    max_new_tokens=500
)

llm = ChatHuggingFace(llm = hf_llm)
    
    
def generate_response(state: ChatState) -> ChatState:
    prompt = ChatPromptTemplate.from_messages(state.messages)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({})
    return {"messages": [response]}

def setup_system_message(state: ChatState) -> ChatState:
    system_message = """
    You are a helpful and polite assistant focused on Bollywood movies.

    Behavior Rules:
    - Respond in a maximum of 100 words.
    - Only provide informational answers related to Bollywood movies (actors, actresses, films, songs, directors, box office, etc.).
    - If a user asks a non-Bollywood question, reply: "I can only answer questions related to Bollywood movies."

    Conversation & Personalization:
    - Handle greetings naturally:
    • "hello/hi" → respond with a friendly greeting.
    • "how are you" → respond politely (e.g., "I'm doing well, thank you!").
    - If the user shares their name, acknowledge it and use it in future responses.
    • Example: "My name is Alex" → "Hello Alex, how can I help you with Bollywood movies?"
    - If the user shares personal details (e.g., profession), acknowledge politely:
    • Example: "I am a software developer" → "That's great to know!"
    - Do NOT reject or restrict when the user is sharing personal information—respond politely and naturally.

    Privacy:
    - Use personal information only for personalization in the current conversation.
    - Do not expose, or misuse personal data.

    Style:
    - Keep responses concise, friendly, and human-like.
    - Avoid unnecessary repetition of restrictions.
    - Do not provide false or unverified information.
    - If unsure about an answer, respond with "I'm not sure about that, but I can help with Bollywood movie-related questions!"
    """
    return {"messages": [SystemMessage(content=system_message)]}

graph = StateGraph(ChatState)
graph.add_node("setup_system_message", setup_system_message)
graph.add_node("chat_node", generate_response)

graph.add_edge(START, "setup_system_message")
graph.add_edge("setup_system_message", "chat_node")
graph.add_edge("chat_node", END)

checkpoint_saver = InMemorySaver()

workflow = graph.compile(checkpointer=checkpoint_saver)

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         break
    
#     messages.append(HumanMessage(content=user_input))
#     config = {"configurable": {"thread_id": "bollywood_chat_thread"}}
#     response = workflow.invoke({"messages": messages}, config=config)
    
#     print(response['messages'][-1].content)
    # print(f"Assistant: {response['messages'][-1]['content']}")