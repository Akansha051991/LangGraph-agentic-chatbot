import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** Page Config & Styling **********************
st.set_page_config(page_title="LangGraph Agent", page_icon="🤖", layout="wide")

# Custom CSS for a more polished look
st.markdown("""
    <style>
    /* Main background and font tweaks */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
        border-right: 1px solid #e0e0e0;
    }

    /* Modern button styling */
    .stButton>button {
        border-radius: 10px;
        height: 3em;
        transition: all 0.2s ease-in-out;
    }
    
    /* Status box styling */
    .stStatus {
        border-radius: 15px;
        border: 1px solid #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True) # <-- Fixed the argument name here

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

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    raw_threads = retrieve_all_threads()
    st.session_state['chat_threads'] = raw_threads[::-1] 

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

with st.sidebar:
    st.title("🤖 AI Assistant")
    st.caption("Powered by LangGraph & OpenAI")
    
    if st.button('➕ Start New Chat', use_container_width=True, type="primary"):
        reset_chat()

    st.divider()
    st.subheader('📜 History')
    
    # Loop through conversation threads
    for thread_id in st.session_state['chat_threads']:
        is_active = thread_id == st.session_state['thread_id']
        # Highlight active thread using the 'type' parameter
        btn_type = "primary" if is_active else "secondary"
        
        if st.button(f"💬 {str(thread_id)[:8]}...", key=f"btn_{thread_id}", 
                     use_container_width=True, type=btn_type):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)
            
            temp_messages = []
            for msg in messages:
                role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages
            st.rerun()

# **************************************** Main UI ************************************

# **************************************** Main UI ************************************

st.title("Chat Interface")
st.info(f"Currently viewing Thread: **{st.session_state['thread_id']}**", icon="🆔")

# Display conversation history
for message in st.session_state['message_history']:
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
                
                # We use stream_mode="messages" to see every step of the graph
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages"
                ):
                    # 1. Handle Tool Calls (The Agent deciding to use a tool)
                    if isinstance(message_chunk, AIMessage) and message_chunk.tool_calls:
                        for tool_call in message_chunk.tool_calls:
                            status_container.update(label=f"Using tool: **{tool_call['name']}**", state="running")
                            status_container.write(f"⚙️ Action: `{tool_call['name']}` with args: `{tool_call['args']}`")
                    
                    # 2. Handle Tool Outputs (The result coming back from the tool)
                    # This tells the user what the tool actually found!
                    from langchain_core.messages import ToolMessage
                    if isinstance(message_chunk, ToolMessage):
                        status_container.write(f"✅ Tool Result: `{str(message_chunk.content)[:100]}...`")

                    # 3. Handle Final Text Content
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        if not has_started_typing:
                            status_container.update(label="✅ Response ready", state="complete", expanded=False)
                            has_started_typing = True
                        yield message_chunk.content

            # Capture the full response to save to history
            full_response = st.write_stream(ai_only_stream)

    # Save to history and refresh
    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
    st.rerun()