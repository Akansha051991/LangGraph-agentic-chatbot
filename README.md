# 🤖 LangGraph Agentic Chatbot with SQL Persistence

An advanced AI Assistant architecture that leverages **LangGraph** for complex state management, **Streamlit** for a modern UI, and **LangSmith** for full-cycle observability. 

This project demonstrates a production-ready approach to building agentic workflows with persistent memory and real-time tool integration.

---

## 🌟 Key Features

* **Agentic Workflow:** Uses a cyclic graph to handle tool-calling loops (Search, Finance, & Math).
* **Multi-Threaded Persistence:** Integrated `SqliteSaver` allows for multiple independent conversation threads stored in a local SQLite database.
* **Real-Time Streaming:** A smooth, ChatGPT-like interface with transparent "Thought" blocks using Streamlit's `st.status`.
* **Enterprise Observability:** Fully integrated with **LangSmith** for tracing, debugging, and performance monitoring.
* **Secure Architecture:** Comprehensive use of environment variables (`.env`) to protect sensitive API keys.

---

## 🛠️ Tech Stack

* **Logic:** [LangGraph](https://www.langchain.com/langgraph) & [LangChain](https://www.langchain.com/)
* **Frontend:** [Streamlit](https://streamlit.io/) (Custom CSS & Multi-threading)
* **Database:** SQLite (SQLAlchemy / LangGraph Checkpointers)
* **Observability:** LangSmith
* **LLM:** OpenAI GPT-4o

---

## 📂 Project Structure

```text

├── langgraph_tool_backend.py      # Core agent logic & state graph
├── streamlit_frontend_threading.py # UI & multi-threaded execution
├── requirements.txt               # Project dependencies
├── .env.example                   # API key template
└── .gitignore                     # Files to exclude from Git

```
## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/LangGraph-agentic-chatbot.git](https://github.com/YOUR_USERNAME/LangGraph-agentic-chatbot.git)
cd LangGraph-agentic-chatbot

```
### 2. Install Dependencies 
pip install -r requirements.txt


### 3. Setup Environment Variables
Create a .env file in the root directory. Add your credentials (see .env.example for reference):
* `OPENAI_API_KEY`=your_openai_key
* `ALPHA_VANTAGE_API_KEY`=your_alpha_vantage_key
* `STOCK_PRICE_URL`=[https://www.alphavantage.co/query](https://www.alphavantage.co/query)

# LangSmith Configuration
* `LANGSMITH_TRACING`=true
* `LANGSMITH_API_KEY`=your_langsmith_key
* `LANGSMITH_PROJECT`="LangGraph-Chatbot"

### 4. Run the Application
Start the Streamlit server to launch the interface:
   streamlit run streamlit_frontend_threading.py

### 5. Verify Observability
After interacting with the bot, visit your LangSmith dashboard to see the execution traces:

https://eu.smith.langchain.com/



## 🔍 Architecture & Observability
By using LangGraph, the agent can loop back and correct itself if a tool call fails. Every "thought" and "action" is logged via LangSmith, providing full transparency into the LLM's reasoning process, token usage, and latency.# LangGraph-agentic-chatbot

A brief description of what this project does and who it's for.
