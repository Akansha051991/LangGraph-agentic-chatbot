import os
import sqlite3
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated

# LangGraph Core
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage

# Cloud LLM: Groq (Replacement for Ollama)
from langchain_groq import ChatGroq

# Tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper, OpenWeatherMapAPIWrapper
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun

load_dotenv()

# 1. INITIALIZE CLOUD LLM (Groq)
# We use llama-3.1-8b-instant for the highest free-tier rate limits
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. TOOLS SETUP
# -------------------
search_tool = TavilySearchResults(max_results=2)

wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

weather_api = OpenWeatherMapAPIWrapper()
weather_tool = OpenWeatherMapQueryRun(api_wrapper=weather_api)

# Scraper-based YouTube tool (No API Key needed)
youtube_tool = YouTubeSearchTool()

tools = [search_tool, wiki_tool, weather_tool, youtube_tool]

# Bind tools to Groq
llm_with_tools = llm.bind_tools(tools)

# 3. STATE DEFINITION
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. NODES
# -------------------
def chat_node(state: ChatState):
    """Processes chat and triggers tools if needed."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# 5. PERSISTENCE (SQLite)
@st.cache_resource
def get_checkpointer():
    """Shared database connection for message history."""
    conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
    return SqliteSaver(conn=conn)

checkpointer = get_checkpointer()

# 6. GRAPH CONSTRUCTION
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# 7. SIDEBAR HELPER
# -------------------
def retrieve_all_threads():
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
    except Exception:
        return []
    return list(all_threads)