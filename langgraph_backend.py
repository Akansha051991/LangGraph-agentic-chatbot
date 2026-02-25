from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
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

#checkpointer 
checkpointer =InMemorySaver()
graph =StateGraph(ChatState)

#add nodes 
graph.add_node('chat_node', chat_node)

#add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot= graph.compile(checkpointer=checkpointer)


