import os
import sqlite3
import uuid
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated

# LangGraph Core
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, trim_messages

# Cloud LLM: Groq
from langchain_groq import ChatGroq

# Tools
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, OpenWeatherMapAPIWrapper

load_dotenv()

# 1. INITIALIZE CLOUD LLM
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=GROQ_API_KEY,
    temperature=0,
    max_retries=2,
    timeout=60
)

# 2. TOOLS SETUP
@tool
def youtube_search(query: str):
    """Search YouTube. Input: search query string. Returns video URLs."""
    from langchain_community.tools import YouTubeSearchTool
    limit_query = f"{query}, 2"
    raw = YouTubeSearchTool().run(limit_query)
    clean_list = raw.replace("'/watch?v=", "'https://www.youtube.com/watch?v=")
    return clean_list.replace("', '", "'\n\n'")

@tool
def get_weather(location: str):
    """Get current weather for a location. Input: 'City, Country' string."""
    try:
        weather_api = OpenWeatherMapAPIWrapper()
        return weather_api.run(location)
    except Exception:
        return f"Could not find weather for '{location}'. Please try a different city name."

search_tool = TavilySearchResults(max_results=2)
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

tools = [search_tool, wiki_tool, get_weather, youtube_search] 

# bind_tools with tool_choice="auto" helps Groq parse function calls better
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

# 3. STATE DEFINITION
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. NODES
def chat_node(state: ChatState):
    messages = state["messages"]
    trimmed_messages = trim_messages(
        messages,
        max_tokens=5000,
        strategy="last",
        token_counter=len, 
        include_system=True,
    )
    system_instruction = SystemMessage(content=(
    "You are a helpful research assistant. "
    "1. Answer the user's question directly and concisely. "
    "2. If a tool returns an error or no results, do not keep retrying the same tool. "
    "   Explain the situation to the user instead. "
    "3. Once you have enough information to answer, STOP calling tools and provide the final response. "
    "4. YouTube links: * [Title](URL) on new lines."
))
    messages_with_instruction = [system_instruction] + trimmed_messages
    try:
        response = llm_with_tools.invoke(messages_with_instruction)
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"⚠️ Groq API Error: {str(e)}")]}

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

# 7. SIDEBAR HELPERS
def retrieve_all_threads():
    unique_threads = []
    try:
        for checkpoint in checkpointer.list(None):
            t_id = checkpoint.config["configurable"]["thread_id"]
            if t_id not in unique_threads:
                unique_threads.append(t_id)
    except Exception:
        return []
    return unique_threads

# 8. DATABASE MANAGEMENT
def clear_all_history():
    """Wipes all LangGraph-related tables dynamically."""
    try:
        conn = sqlite3.connect(database="chatbot.db", timeout=10)
        cursor = conn.cursor()
        
        # 1. Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # 2. Clear checkpoint/write tables
        for table in tables:
            if table.startswith("checkpoint") or table == "writes":
                cursor.execute(f"DELETE FROM {table}")
        
        conn.commit()
        conn.close()
        
        # 3. Reset the cache
        get_checkpointer.clear() 
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False