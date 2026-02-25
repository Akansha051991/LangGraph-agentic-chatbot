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
from langchain_core.messages import SystemMessage

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
# This checks Streamlit Cloud Secrets FIRST, then falls back to your local .env
# langgraph_tool_backend.py
import streamlit as st
import os
from langchain_groq import ChatGroq

# Safer way to grab the key from either Streamlit Cloud or Local .env
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY. Please set it in Streamlit Secrets or .env file.")

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    api_key=GROQ_API_KEY  # Explicitly pass the variable here
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
from langchain_core.messages import SystemMessage

def chat_node(state: ChatState):
    """Processes chat and triggers tools if needed."""
    messages = state["messages"]
    
    # Updated Instruction: Be helpful and share results, don't be a critic!
    system_instruction = SystemMessage(content=(
        "You are a helpful research assistant. When a user asks for videos or searches: "
        "1. Execute the tool. "
        "2. Directly provide the titles and URLs found. "
        "3. Do not complain about 'lack of specifications' if the user only asked to find videos. "
        "4. Only use URLs exactly as provided by the tools."
    ))
    
    messages_with_instruction = [system_instruction] + messages
    response = llm_with_tools.invoke(messages_with_instruction)
    
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
