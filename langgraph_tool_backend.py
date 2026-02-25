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
# ADDED: AIMessage and trim_messages to imports
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, trim_messages

# Cloud LLM: Groq
from langchain_groq import ChatGroq

# Tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper, OpenWeatherMapAPIWrapper
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun

load_dotenv()

# 1. INITIALIZE CLOUD LLM (Groq)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY. Please set it in Streamlit Secrets or .env file.")

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=GROQ_API_KEY,
    temperature=0,
    max_retries=2,
    timeout=60
)

# 2. TOOLS SETUP
search_tool = TavilySearchResults(max_results=2)
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)
weather_api = OpenWeatherMapAPIWrapper()
weather_tool = OpenWeatherMapQueryRun(api_wrapper=weather_api)
youtube_tool = YouTubeSearchTool()

tools = [search_tool, wiki_tool, weather_tool, youtube_tool]
llm_with_tools = llm.bind_tools(tools)

# 3. STATE DEFINITION
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. NODES (Corrected with Trimming and Error Handling)
# ------------------------------------------------------
def chat_node(state: ChatState):
    """Processes chat with context window protection."""
    messages = state["messages"]
    
    # 1. Trim messages to avoid Groq Rate/Token limits (TPM)
    # We use len as a simple counter; 5000 chars is roughly 1000-1200 tokens
    trimmed_messages = trim_messages(
        messages,
        max_tokens=5000,
        strategy="last",
        token_counter=len, 
        include_system=True,
    )

    # 2. Personality & Strict URL Rules
    system_instruction = SystemMessage(content=(
        "You are a helpful research assistant. When a user asks for videos or searches: "
        "1. Execute the tool. "
        "2. Directly provide the titles and URLs found. "
        "3. Provide YouTube links as [Title](URL). "
        "4. Do not complain about lack of specs; just share what the tool found. "
        "5. ONLY use URLs exactly as provided by the tools. Do not invent links."
    ))
    
    messages_with_instruction = [system_instruction] + trimmed_messages
    
    try:
        # 3. Invoke the model
        response = llm_with_tools.invoke(messages_with_instruction)
        return {"messages": [response]}
    except Exception as e:
        # 4. Catch Groq API Status Errors gracefully
        error_msg = f"⚠️ Groq API Error: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}

tool_node = ToolNode(tools)

# 5. PERSISTENCE (SQLite)
@st.cache_resource
def get_checkpointer():
    conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
    return SqliteSaver(conn=conn)

checkpointer = get_checkpointer()

# 6. GRAPH CONSTRUCTION
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# 7. SIDEBAR HELPER
def retrieve_all_threads():
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
    except Exception:
        return []
    return list(all_threads)