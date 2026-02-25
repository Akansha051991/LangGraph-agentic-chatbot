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
from langchain_core.tools import tool
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
@tool
def youtube_search(query: str):
    """Searches YouTube and returns a clean list of full URLs."""
    from langchain_community.tools import YouTubeSearchTool
    # Logic: "query, number" limits the results. 
    # Here we limit it to 2 results.
    limit_query = f"{query}, 2"
    # This runs the search and gives us the raw list as a string
    raw = YouTubeSearchTool().run(limit_query)
    
    # Convert relative paths to full URLs
    clean_list = raw.replace("'/watch?v=", "'https://www.youtube.com/watch?v=")
    
    # Return with clean formatting
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

# 4. NODES (Corrected with Trimming and Error Handling)
# ------------------------------------------------------
def chat_node(state: ChatState):
    """Processes chat with context window protection."""
    messages = state["messages"]
    
    # 1. Trim messages to avoid Groq Rate/Token limits (TPM)
    trimmed_messages = trim_messages(
        messages,
        max_tokens=5000,
        strategy="last",
        token_counter=len, 
        include_system=True,
    )

    # 2. Personality & Strict URL Rules 
    system_instruction = SystemMessage(content=(
    "You are a helpful research assistant. "
    
   # Rule 1: Directness (Fixes the "Capital City" distraction)
    "1. Answer the user's question directly for the specific location they provided. "
    "Do not mention capital cities unless the user specifically asks for them."

     # Rule 2: Weather Data (Ensures it actually reports the numbers)
    "2. When using the weather tool, you MUST include the temperature and weather conditions "
     "in your final answer. Example: 'The weather in Tokyo is 15°C with clear skies.'"

    # --- YouTube Formatting ---
    "3. When providing YouTube links, you MUST put each link on a NEW LINE. "
    "Format them as a bulleted list: * [Video Title](URL)"
))
    # Align this exactly with the system_instruction above
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
    unique_threads = []
    try:
        # LangGraph's checkpointer.list() returns checkpoints 
        # from newest to oldest by default.
        for checkpoint in checkpointer.list(None):
            t_id = checkpoint.config["configurable"]["thread_id"]
            
            # Only add the thread_id if we haven't seen it yet.
            # This ensures each thread appears only once and 
            # stays in its 'most recent' position.
            if t_id not in unique_threads:
                unique_threads.append(t_id)
                
    except Exception as e:
        print(f"Error retrieving threads: {e}")
        return []
        
    return unique_threads  # This is now sorted: Newest -> Oldest
# 8. DATABASE MANAGEMENT
def clear_all_history():
    """Wipes the SQLite database to clear all threads."""
    try:
        # Use the same database name you used in get_checkpointer()
        conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
        cursor = conn.cursor()
        # These are the standard LangGraph persistence tables
        cursor.execute("DELETE FROM checkpoints")
        cursor.execute("DELETE FROM checkpoint_blobs")
        cursor.execute("DELETE FROM checkpoint_writes")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error clearing history: {e}")
        return False