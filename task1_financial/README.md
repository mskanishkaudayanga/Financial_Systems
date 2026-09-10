# Financial AI Equity Research Pipeline (Task 1)

An end-to-end LLM-powered equity research pipeline combining quantitative technical indicators, financial news sentiment extraction, and structured trading recommendations.

---

## 🏗 System Architecture & Directory Organization

```
task1_financial/
├── notebook.ipynb         # Interactive execution & demonstration layer
├── src/                   # Core reusable business logic & modules
│   ├── config.py          # Centralized environment-driven configuration
│   ├── data/              # Ingestion layer (market OHLCV & news scraping)
│   ├── features/          # Quantitative feature engineering (technical indicators)
│   ├── llm/               # LLM client abstractions, sentiment analysis & signal generation
│   ├── schemas/           # Pydantic data models for structured output validation
│   └── prompts/           # Isolated prompt templates for LLM tasks
│   └── pipeline.py        # Top-level orchestration pipeline entrypoint
├── tests/                 # Unit test suites for data, indicators, and schemas
├── README.md              # Project documentation & architectural guide
└── requirements.txt       # Project dependencies
```

### Module Responsibilities

- **`src/config.py`**: Manages environment variables using `python-dotenv`. Ensures API keys and secrets are injected via runtime environment variables rather than hardcoded into source code.
- **`src/data/`**: Isolates external API interaction (e.g. `yfinance` market data and news providers), decoupling data fetching from downstream processing.
- **`src/features/`**: Implements quantitative metrics (SMA, RSI, MACD, Bollinger Bands) using vectorized NumPy/Pandas ops to maintain zero external C-library dependencies (e.g. TA-Lib).
- **`src/schemas/`**: Enforces type safety and structured JSON validation via Pydantic for LLM responses and data transfer objects.
- **`src/prompts/`**: Decouples prompt engineering from Python execution code, enabling versioning and modular adjustments to prompt instructions.
- **`src/llm/`**: Encapsulates LLM provider calls, sentiment classification, and trading signal synthesis logic.
- **`src/pipeline.py`**: Clean, functional pipeline coordinator connecting data fetching, indicator calculation, and LLM reasoning.
- **`notebook.ipynb`**: Minimal presentation layer demonstrating the pipeline workflow without embedding core business logic in notebook cells.

---

## 🔒 Configuration & Environment Variables

All sensitive values (such as LLM API keys) and configurable defaults must be supplied via environment variables.

Copy `.env.example` (or create a `.env` file locally):
```bash
DEFAULT_TICKER=AAPL
LOOKBACK_YEARS=2
NEWS_HEADLINES_LIMIT=10
LLM_API_KEY=your_api_key_here
LLM_MODEL_NAME=gpt-4o-mini
LOG_LEVEL=INFO
```

> **Note**: Never commit `.env` to version control. `.env` is explicitly ignored by `.gitignore`.

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/
```
