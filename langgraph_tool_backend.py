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
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, trim_messages

# Cloud LLM: Groq
from langchain_groq import ChatGroq

# Tools
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper, OpenWeatherMapAPIWrapper
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun

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

@tool
def youtube_search(query: str):
    """Searches YouTube and returns a clean list of full URLs."""
    from langchain_community.tools import YouTubeSearchTool
    limit_query = f"{query}, 2"
    raw = YouTubeSearchTool().run(limit_query)
    clean_list = raw.replace("'/watch?v=", "'https://www.youtube.com/watch?v=")
    return clean_list.replace("', '", "'\n\n'")

# 2. TOOLS SETUP
search_tool = TavilySearchResults(max_results=2)
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)
weather_api = OpenWeatherMapAPIWrapper()
weather_tool = OpenWeatherMapQueryRun(api_wrapper=weather_api)

tools = [search_tool, wiki_tool, weather_tool, youtube_search] 
llm_with_tools = llm.bind_tools(tools)

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
        "1. Answer the user's question directly. "
    
        "2. TOOL USAGE RULES:"
        "   - ONLY use the weather tool if the user explicitly asks about weather, "
        "     forecasts, or current temperatures."
        "   - DO NOT provide weather information for locations mentioned in general "
         "    research (like Kansas or Indiana) unless weather was part of the query."
        "3.   When providing YouTube links, you MUST put each link on a NEW LINE. "
        "     Format YouTube links as bulleted lists: * [Title](URL) on new lines."
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
    # check_same_thread=False is crucial for Streamlit
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

# 8. DATABASE MANAGEMENT (Fixed Table Names & Logic)
def clear_all_history():
    """Wipes all LangGraph-related tables dynamically."""
    try:
        conn = sqlite3.connect(database="chatbot.db", timeout=10)
        cursor = conn.cursor()
        
        # 1. Get all table names currently in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # 2. Delete data from LangGraph specific tables if they exist
        # Common names across versions: 'checkpoints', 'checkpoint_blobs', 'checkpoint_writes', 'writes'
        for table in tables:
            if table.startswith("checkpoint") or table == "writes":
                cursor.execute(f"DELETE FROM {table}")
                print(f"Cleared table: {table}")
        
        conn.commit()
        conn.close()
        
        # 3. Reset the Streamlit checkpointer cache
        get_checkpointer.clear() 
        
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False