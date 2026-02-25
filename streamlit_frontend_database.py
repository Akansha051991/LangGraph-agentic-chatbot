import streamlit as st
from langgraph_backend_database import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from datetime import datetime

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    
    if thread_id not in st.session_state['chat_threads']:
        # Insert at index 0 so the newest thread is always at the top
        st.session_state['chat_threads'].insert(0, thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    raw_threads = retrieve_all_threads()
    st.session_state['chat_threads'] = raw_threads[::-1]
# Ensure current active thread is in the list and at the top
add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat', use_container_width=True):
    reset_chat()

st.sidebar.markdown("---")
st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads']:
    # Use the thread_id as the key to prevent Streamlit button behavior issues
    if st.sidebar.button(f"Chat: {str(thread_id)[:8]}...", key=f"btn_{thread_id}", use_container_width=True):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        st.rerun() # Refresh UI to show the selected conversation


# **************************************** Main UI ************************************

# loading the conversation history
st.title("Chat Interface")
st.caption(f"Current Thread ID: {st.session_state['thread_id']}")

# Display conversation history

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here....')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

# 2. Generate and stream AI response
    #CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    CONFIG ={
        "configurable": {'thread_id': st.session_state['thread_id']},
        "metadata":{
            "thread_id":st.session_state["thread_id"]
        },
        "run_name":"chat_turn",}
    

     # first add the message to message_history
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())
# 3. Save assistant message to history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})