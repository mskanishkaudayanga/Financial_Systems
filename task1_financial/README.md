# Financial AI Equity Research Pipeline (Task 1)

An end-to-end Financial AI equity research system combining quantitative technical indicators, financial news retrieval, LLM sentiment extraction, and synthesized Buy/Hold/Sell recommendation reasoning.

---

## 🏛 System Architecture & Design Philosophy

The system follows a **Neuro-Symbolic Architecture** that explicitly decouples deterministic quantitative calculations (Python/Pandas/NumPy) from qualitative AI reasoning (LLM inference).

```
                      +----------------------------------+
                      |       yfinance & Environment     |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |    1. Market Data Ingestion      |
                      |   (src/data/market_data.py)      |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      | 2. Quantitative Indicators Math  |
                      | (src/features/technical_ind.py)  |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      | 3. Summary & News Retrieval      |
                      | (src/features/summary.py & news) |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      | 4. Single-Call Batch Sentiment   |
                      |    (src/llm/sentiment.py)        |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      | 5. Recommendation Synthesis      |
                      |     (src/llm/signal.py)          |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |  6. FinancialPipelineResult      |
                      |   (src/schemas/models.py)        |
                      +----------------------------------+
```

### Core Design Principles

1. **Zero Arithmetic Hallucination**: Technical indicators (SMA-50, SMA-200, RSI-14, MACD, Bollinger Bands) are calculated exclusively in Python using vectorized Pandas/NumPy operations. The LLM never calculates indicators itself; it interprets pre-computed metrics.
2. **Single-Call Batch LLM Efficiency**: All 10–15 news headlines are analyzed in **1 single batch LLM API call** ($T=0.0$), reducing latency from 15s to ~1.5s and cutting token overhead by ~80%.
3. **Strict Pydantic V2 Schema Validation**: All external API outputs, sentiment scores, trading signals, and final pipeline payloads are strictly validated using Pydantic schemas (`HeadlineSentiment`, `AggregatedSentiment`, `TradingRecommendation`, `FinancialPipelineResult`).
4. **Defensive Fail-Safe Fallbacks**: Non-critical failures (LLM API timeouts, news API rate limits, corrupt JSON) trigger deterministic fallbacks (`neutral` sentiment, `HOLD` recommendation) derived from Python technical momentum. The pipeline never crashes on third-party API issues.

---

## 🏗 Directory Organization & Module Map

```
task1_financial/
├── .env.example           # Environment configuration template
├── .gitignore             # Version control exclusion rules
├── conftest.py            # Pytest configuration & sys.path root resolver
├── notebook.ipynb         # Interactive demonstration notebook
├── README.md              # Project documentation & architectural guide
├── requirements.txt       # Project dependencies
├── src/                   # Core business logic modules
│   ├── config.py          # Centralized environment-driven configuration
│   ├── pipeline.py        # High-level pipeline orchestrator
│   ├── data/
│   │   ├── market_data.py # OHLCV market data ingestion (yfinance)
│   │   └── news_data.py   # News headline retrieval & normalization
│   ├── features/
│   │   ├── technical_indicators.py # SMA, RSI, MACD, Bollinger Bands (No TA-Lib)
│   │   └── summary.py     # YTD return, 52w bounds, & deterministic momentum signal
│   ├── llm/
│   │   ├── client.py      # OpenAI-compatible REST API client
│   │   ├── sentiment.py   # Single-call batch news sentiment extraction & aggregation
│   │   └── signal.py      # LLM Buy/Hold/Sell trading signal synthesis
│   ├── prompts/
│   │   ├── sentiment.py   # Isolated prompt templates for news sentiment
│   │   └── recommendation.py # Isolated prompt templates for recommendation
│   └── schemas/
│       └── models.py      # Pydantic V2 data validation models
└── tests/                 # Offline unit & integration test suites
    ├── test_data.py
    ├── test_indicators.py
    ├── test_llm_sentiment.py
    ├── test_llm_signal.py
    ├── test_news.py
    ├── test_pipeline.py
    ├── test_schemas.py
    └── test_summary.py
```

---

## 🧩 Module Responsibilities

- **`src/config.py`**: Manages environment settings using `python-dotenv`. Credentials and secrets are injected strictly via environment variables (zero hardcoded keys).
- **`src/data/market_data.py`**: Ingests daily OHLCV bars using `yfinance` with dynamic date ranges (`date.today()`), MultiIndex header flattening, and forward/backward fill missing value imputation.
- **`src/data/news_data.py`**: Ingests financial news headlines, normalizes raw payloads into `{headline, source, published_at, url}`, deduplicates titles, and enforces threshold checks.
- **`src/features/technical_indicators.py`**: Calculates SMA-50, SMA-200, RSI-14 (Wilder's smoothing method), MACD (12, 26, 9), and Bollinger Bands (20, 2) from mathematical first principles using Pandas/NumPy without TA-Lib.
- **`src/features/summary.py`**: Calculates dynamic YTD returns, 52-week price bounds, P/E ratio extraction (without data fabrication), and rule-based deterministic momentum scoring.
- **`src/schemas/models.py`**: Pydantic V2 validation schemas (`HeadlineSentiment`, `AggregatedSentiment`, `TradingRecommendation`, `FinancialPipelineResult`).
- **`src/prompts/`**: Isolated system and user prompt templates mandating structured JSON output and cross-indicator combination reasoning.
- **`src/llm/client.py`**: OpenAI-compatible REST API client wrapper using `requests` with configurable timeout, sampling temperature (`0.0`/`0.1`), and model parameters.
- **`src/llm/sentiment.py`**: Executes single-call batch LLM sentiment extraction over headlines and computes a confidence-weighted aggregate sentiment score.
- **`src/llm/signal.py`**: Synthesizes Buy/Hold/Sell trading recommendations over pre-calculated Python indicators.
- **`src/pipeline.py`**: High-level orchestrator that connects all modules and returns a validated `FinancialPipelineResult`.

---

## 🔒 Configuration & Environment Variables

Create a `.env` file inside `task1_financial/` based on `.env.example`:

```env
# Financial Market Parameters
DEFAULT_TICKER=AAPL
LOOKBACK_YEARS=2

# News Parameters
NEWS_HEADLINES_LIMIT=10

# LLM API Provider Configuration (OpenAI, Groq, OpenRouter, etc.)
LLM_API_KEY=your_llm_api_key_here
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_TEMPERATURE=0.0
LLM_TIMEOUT=15

# Logging
LOG_LEVEL=INFO
```

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite (Offline with Mocks)
```bash
pytest tests/
```

### 3. Run End-to-End Pipeline
```bash
python -m src.pipeline
```

### 4. Interactive Demonstration Notebook
Open `notebook.ipynb` in VS Code or Jupyter and execute all cells.
