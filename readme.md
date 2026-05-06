# Production Agentic Research Assistant

A multi-agent LangGraph system for autonomous research automation — task decomposition, real-time web search synthesis, and self-correcting reasoning. Runs fully local via ChatOllama + Llama 3.1 with zero external API dependency.

## Architecture

```
User Query (Streamlit)
        │
        ▼
   [chatbot node]  ← Llama 3.1 via ChatOllama (local inference)
        │
        ▼
 recursion_guard()
        │
   ┌────┴─────────────────┐
   │                      │
   ▼                      ▼
[tools node]         [reflect node]  ← triggered when recursion_count >= 5
DuckDuckGo search    Reviews + corrects last answer
        │                      │
        ▼                      ▼
   [chatbot node]             END
  (think again)
```

## Key Design

- `recursion_count` in Graph-State increments every chatbot call
- `recursion_guard()` replaces standard `tools_condition` — checks count before routing
- At ceiling (5), routes to `reflect` node instead of looping forever
- `reflect` node prompts LLM to review its own answer before ending
- All inference runs locally — no OpenAI API, no cost, no rate limits

## Tech Stack

- **LangGraph** — StateGraph, conditional edges, Graph-State management
- **ChatOllama + Llama 3.1** — local LLM inference
- **DuckDuckGo** — free real-time web search via langchain_community
- **Streamlit** — chat UI with session-state message history
- **FastAPI + Docker + CI/CD** — production deployment

## Project Structure

```
src/
├── agent.py        # StateGraph, State, nodes, recursion_guard, create_agent()
├── tools.py        # DuckDuckGo search tool
└── application.py  # Streamlit chat interface
requirements.txt
Dockerfile
.gitignore
```

## Quickstart

```bash
# 1. Install Ollama and pull Llama 3.1
ollama pull llama3.1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run src/application.py

# Or via Docker
docker build -t agentic-researcher .
docker run -p 8501:8501 agentic-researcher
```

## Graph Flow

```
START → chatbot → recursion_guard → tools → chatbot (loop with DuckDuckGo)
                                  → reflect → END (self-correction)
                                  → END (answer complete)
```
