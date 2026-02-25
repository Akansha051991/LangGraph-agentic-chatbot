from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver #Implementation of LangGraph CheckpointSaver that uses SQLite DB (both sync and async, via aiosqlite)
from dotenv import load_dotenv
import sqlite3
load_dotenv()

llm = ChatOpenAI()
#defining state
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
#function
def chat_node(state:ChatState):
    #1.take user query from state
    messages= state['messages']#extracting messages from state
    #2.send to llm - #for that we need to define an llm
    response= llm.invoke(messages) 
    #3.storing response in state
    return {'messages':[response]}
################ Initializes a thread-safe SQLite checkpointer to persist and retrieve the chatbot's state (memory) using a local database file#################################

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
#checkpointer 
checkpointer =SqliteSaver(conn=conn)

graph =StateGraph(ChatState)
#add nodes 
graph.add_node('chat_node', chat_node)
#add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot= graph.compile(checkpointer=checkpointer)
def retrieve_all_threads():

    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return(list(all_threads))
