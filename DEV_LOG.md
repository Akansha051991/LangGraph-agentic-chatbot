# 📓 Developer Log: Agentic Research Assistant

This log tracks the technical evolution, hurdles, and architectural decisions made during the development of this project.

## 🛠️ Phase 1: The Foundation (Backend & UI)
- **Goal**: Transition from a basic script to a functional web app.
- **Action**: Built the core logic using Python and integrated **Groq LPU** for sub-second inference.
- **UI Upgrade**: Implemented **Streamlit** to handle user inputs.
- **Streaming**: Added real-time token streaming to ensure the UX felt responsive, even during complex tool calls.

## 🧠 Phase 2: Orchestration & Statefulness
- **Graph Logic**: Migrated the backend to **LangGraph** to handle cyclic "Think-Act-Observe" loops.
- **Persistence**: Integrated a **SQLite checkpointer**. 
    - *Lesson*: This allowed for "Resumable Chats" (Thread IDs), turning the bot from a one-off calculator into a persistent assistant that remembers context.
- **Observability**: Connected **LangSmith** to the pipeline.
    - *Lesson*: This was vital for debugging "hallucinations" in tool selection and monitoring token usage.

## 🔧 Phase 3: Tool Integration & Prompt Engineering
- **Tooling**: Added `TavilySearch`, `Wikipedia`, `YouTubeSearch`, and `OpenWeatherMap`.
- **Hurdle: The "Capital City" Distraction**: 
    - *Problem*: Small models (Llama-3.1-8B) tended to get distracted by geography trivia instead of fetching weather data.
    - *Fix*: Refined the **System Instructions** to prioritize direct data delivery over conversational filler.
- **Parallel Execution**: Optimized the `ToolNode` to trigger multiple tools simultaneously (e.g., getting weather and a video in one turn).

## 🚀 Phase 4: Git & Environment Debugging
- **PostBuffer Issue**: Encountered a "hang" during Git commits due to large file indexing (SQLite DB).
- **Resolution**:
    - Increased `http.postBuffer` to `524288000`.
    - Added Git to the System PATH to allow for CLI-based troubleshooting.
    - Established a `.gitignore` to keep large databases and `.venv` files out of the repository.

## 🔮 Future Roadmap: The MCP Transition
The next major architectural shift will be the implementation of **Model Context Protocol (MCP)**.
- **Standardization**: Moving away from custom tool wrappers to a universal MCP server.
- **Decoupling**: Separating the "Host" (this app) from the "Servers" (the tools) for better scalability and modularity.
