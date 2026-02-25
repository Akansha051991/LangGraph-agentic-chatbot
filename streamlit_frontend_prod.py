import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import datetime
import re

# **************************************** Page Config & Styling **********************
st.set_page_config(page_title="LangGraph Agent", page_icon="🤖", layout="wide")

# Custom CSS for a more polished look
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
        border-right: 1px solid #e0e0e0;
    }
    .stButton>button {
        border-radius: 10px;
        height: 3em;
        transition: all 0.2s ease-in-out;
    }
    .stStatus {
        border-radius: 15px;
        border: 1px solid #f0f2f6;
    }
    /* Styles the session caption */
    .session-info {
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# **************************************** Utility Functions *************************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].insert(0, thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def get_thread_label(thread_id):
    """Generates a human-readable label for the sidebar buttons."""
    messages = load_conversation(thread_id)
    if messages:
        # Look for the first human message to use as a title
        first_user_msg = next((m.content for m in messages if isinstance(m, HumanMessage)), None)
        if first_user_msg:
            words = first_user_msg.split()
            # Takes first 4 words and capitalizes them
            title = " ".join(words[:4]).title()
            return f"{title}..." if len(words) > 4 else title
    return f"New Chat {str(thread_id)[:5]}"

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    raw_threads = retrieve_all_threads()
    # If database is empty, start with current thread
    st.session_state['chat_threads'] = raw_threads if raw_threads else []

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

with st.sidebar:
    st.title("🤖 AI Assistant")
    # Locate this in your Sidebar UI section
    st.caption("Powered by LangGraph & Groq (Llama 3.1)")
    
    if st.button('➕ Start New Chat', use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()
        # --- ADDED: Clear History Button ---
    if st.button('🗑️ Clear All History', use_container_width=True):
        from langgraph_tool_backend import clear_all_history
        if clear_all_history():
            st.session_state['chat_threads'] = []
            st.session_state['message_history'] = []
            # Generate a fresh thread ID so we aren't on a deleted one
            st.session_state['thread_id'] = str(uuid.uuid4()) 
            st.toast("Database Wiped Clean!")
            st.rerun()

    st.divider()
    st.subheader('📜 History')
    
    # Loop through conversation threads with Smart Labels
    for thread_id in st.session_state['chat_threads']:
        is_active = thread_id == st.session_state['thread_id']
        btn_type = "primary" if is_active else "secondary"
        
        # Use our helper function for the label
        label = get_thread_label(thread_id)
        
        if st.button(f"💬 {label}", key=f"btn_{thread_id}", 
                     use_container_width=True, type=btn_type):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)
            
            temp_messages = []
            for msg in messages:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                    temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages
            st.rerun()

# **************************************** Main UI ************************************
st.title("Chat Interface")

# Cleaner Session ID display
st.markdown(f"<div class='session-info'>🧵 Session: <code>{st.session_state['thread_id']}</code></div>", unsafe_allow_html=True)

# Display conversation history
for message in st.session_state['message_history']:
    if message['content']: # Only display if there is text content
        with st.chat_message(message['role']):
            st.markdown(message['content'])

user_input = st.chat_input('How can I help you today?')

if user_input:
    # Add user message to UI
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    with st.chat_message("assistant"):
        with st.status("🔍 Thinking...", expanded=True) as status_container:
            
            def ai_only_stream():
                has_started_typing = False
                
                for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={
                    "configurable": {"thread_id": st.session_state['thread_id']},
                    "recursion_limit": 10  # This kills the loop after 10 steps
                }, 
                stream_mode="messages",
):
                    # 1. Tool Calls
                    if isinstance(message_chunk, AIMessage) and message_chunk.tool_calls:
                        for tool_call in message_chunk.tool_calls:
                            status_container.update(label=f"Running Tool: **{tool_call['name']}**", state="running")
                            status_container.write(f"⚙️ Calling: `{tool_call['name']}`with args: `{tool_call['args']}`")
                    
                    # 2. Tool Outputs
                    if isinstance(message_chunk, ToolMessage):
                        status_container.write(f"✅ Tool results integrated.")

                    # 3. Final Text
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        if not has_started_typing:
                            status_container.update(label="✅ Complete", state="complete", expanded=False)
                            has_started_typing = True
                        yield message_chunk.content

            full_response = st.write_stream(ai_only_stream)
# Check for YouTube links to embed
        yt_pattern = r'(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+|https?://youtu\.be/[\w-]+)'
        yt_links = re.findall(yt_pattern, full_response)
    
        if yt_links:
            for link in yt_links:
                st.video(link) # This puts the actual video player in the chat
    # Save to history
    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
    
    # Ensure current thread is in the history list
    if st.session_state['thread_id'] not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].insert(0, st.session_state['thread_id'])
        
    st.rerun()