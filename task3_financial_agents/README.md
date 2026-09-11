# Task 3: Financial Multi-Agent Research System Foundation

This project establishes the foundation for a specialized **Multi-Agent Financial Research System** built on **LangChain** and **LangGraph**, combined with custom Python infrastructure for deterministic computation and observability.

---

## 📂 Project Directory Structure

```text
task3_financial_agents/
├── notebook.ipynb         # Interactive demonstration & evaluation notebook
├── src/
│   ├── __init__.py        # Top-level package initialization
│   ├── config.py          # Centralized configuration management (python-dotenv)
│   ├── tools/             # Financial, news, & technical analysis tools
│   ├── schemas/           # Pydantic schemas & state representations
│   ├── agents/            # Specialized agent node definitions
│   ├── workflows/         # LangGraph state graph & routing logic
│   ├── memory/            # Checkpointers & state persistence
│   └── observability/     # Trace logging & audit trail infrastructure
├── cache/                 # Local filesystem cache for data & LLM responses
├── agent_trace.jsonl      # Step-by-step agent execution audit log
├── CITATIONS.md           # AI assistance log & literature references
├── README.md              # Project documentation (this file)
├── requirements.txt       # Project dependencies
├── .env.example           # Template environment configuration
└── .gitignore             # Git exclusion rules
```

---

## 🔑 Configuration Setup (`src/config.py`)

Configuration management uses `python-dotenv` to load environment variables without exposing secrets.

### Environment Variable Keys (`.env`)

- `LLM_PROVIDER`: Provider name (e.g. `openai`, `google`, `anthropic`). Default: `openai`.
- `LLM_MODEL_NAME`: Target model (e.g. `gpt-4o-mini`, `gemini-1.5-flash`). Default: `gpt-4o-mini`.
- `LLM_API_KEY`: API authentication key.
- `LLM_BASE_URL`: API gateway / base endpoint. Default: `https://api.openai.com/v1`.
- `LLM_TEMPERATURE`: LLM sampling temperature. Default: `0.0`.
- `CACHE_DIR`: Directory for data caching. Default: `cache`.
- `TRACE_FILE`: File path for JSONL trace output. Default: `agent_trace.jsonl`.
- `LOG_LEVEL`: Logging verbosity (`INFO`, `DEBUG`). Default: `INFO`.
- `MAX_RECURSION_LIMIT`: Maximum state graph recursion limit. Default: `25`.

---

## 🏗️ Architectural Framework Breakdown

### 1. Directory Responsibilities

- **`src/tools/`**: Houses tool implementations (e.g., market data fetcher, technical indicator calculator, financial news retriever) wrapped as LangChain tools with strict input schemas.
- **`src/schemas/`**: Contains Pydantic models for data validation, structured output schemas, and TypedDict / Pydantic definitions for the global agent graph state (`AgentState`).
- **`src/agents/`**: Defines prompt templates and agent nodes (e.g., Quantitative Agent, News Sentiment Agent, Synthesis Agent) that run LLMs with specific tool bindings.
- **`src/workflows/`**: Compiles the LangGraph `StateGraph`, defining nodes, directed edges, conditional entry points, and state transition logic.
- **`src/memory/`**: Manages state checkpointing, conversation memory buffers (`MemorySaver`), and multi-turn state retention.
- **`src/observability/`**: Manages runtime tracing, streaming output captures, token metrics, and appending step events to `agent_trace.jsonl`.
- **`cache/`**: Persists market data and intermediate calculations to minimize redundant API calls.

### 2. LangChain Components
LangChain provides standard primitives and abstractions:
- **`BaseTool` / `@tool` decorator** in `src/tools/`
- **`ChatOpenAI` / `BaseChatModel`** bindings in `src/agents/`
- **`PromptTemplate` / `ChatPromptTemplate`** in `src/agents/`
- **Structured output parsers (`with_structured_output`)** in `src/schemas/` & `src/agents/`

### 3. LangGraph Components
LangGraph orchestrates agent control flow and state management:
- **`StateGraph`** in `src/workflows/`
- **Graph State Channel (`AgentState`)** in `src/schemas/`
- **Conditional Edges (`add_conditional_edges`)** for routing based on intermediate decisions
- **Checkpointers (`MemorySaver`, `SqliteSaver`)** in `src/memory/`

### 4. Normal Python Infrastructure
Standard Python modules handle system-level logic:
- **`src/config.py`**: Configuration loading via `python-dotenv` and standard `dataclass`.
- **`src/observability/`**: Custom JSONL trace logging, file IO, and timing logic.
- **`cache/`**: Filesystem disk caching using Python built-in `pathlib` and `json`/`pickle`.
- **Numerical Computations**: Pandas and NumPy deterministic data transformations in `src/tools/`.
