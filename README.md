# Senior ML Engineer Technical Assessment Submission

This repository contains the completed solutions for the **CDAZZDEV Senior ML Engineer Technical Assessment**.

---

## 📂 Repository Structure

```text
CDAZZDEV-MLE-[YourName]/
├── README.md               # Root repository guide (this file)
├── CITATIONS.md            # AI Assistance Log & Open-Source Code References
├── REFLECTION.md           # Technical Reflection & Engineering Analysis (<600 words)
│
├── task1_financial/        # Task 1: Financial AI Equity Research Pipeline
│   ├── README.md           # Detailed architecture, setup, & module guide for Task 1
│   ├── notebook.ipynb      # Interactive demonstration notebook (saved cell outputs)
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # Environment variable template
│   ├── conftest.py         # Pytest setup & sys.path resolver
│   ├── src/                # Modular business logic
│   │   ├── config.py       # Configuration management (python-dotenv)
│   │   ├── pipeline.py     # End-to-end pipeline orchestrator
│   │   ├── data/           # Market OHLCV & news data ingestion
│   │   ├── features/       # Technical indicators (SMA, RSI, MACD, BB) & summary
│   │   ├── llm/            # REST API client, single-call batch sentiment, signal synthesis
│   │   ├── prompts/        # Prompt templates (escaped JSON)
│   │   └── schemas/        # Pydantic V2 validation schemas
│   └── tests/              # 44 passing offline unit & integration tests
│
├── task2_genai/            # Task 2 (Sub-folder ready for Task 2 attempt)
└── task3_agentic/          # Task 3 (Sub-folder ready for Task 3 attempt)
```

---

## 🎯 Task 1 Overview: Financial AI Equity Research Pipeline

The **Task 1** system is an end-to-end, neuro-symbolic equity research pipeline that combines deterministic quantitative indicators (calculated from first principles using Pandas/NumPy without TA-Lib) with Large Language Model (LLM) qualitative reasoning.

### Key Highlights

1. **Zero Arithmetic Hallucination**: SMA-50, SMA-200, RSI-14 (Wilder's smoothing), MACD (12, 26, 9), and Bollinger Bands (20, 2) are calculated purely in Python.
2. **Single-Call Batch LLM Efficiency**: All 10–15 news headlines are analyzed in **1 single batch API call** ($T=0.0$), reducing latency to ~1.5s and token overhead by ~80%.
3. **Strict Pydantic V2 Validation**: End-to-end schema enforcement across external API outputs, sentiment scores, and recommendation signals.
4. **Defensive Fail-Safe Fallbacks**: Multi-tier fallbacks prevent pipeline crashes on third-party API rate limits, timeouts, or corrupt JSON.
5. **Comprehensive Test Suite**: **44 / 44 tests passing** with 100% offline mocking (`pytest tests/`).

For detailed documentation, architectural diagrams, and setup instructions, please see [task1_financial/README.md](file:///c:/Users/mskan/Desktop/Technical_Assyment_Cdazzdev/task1_financial/README.md).

---

## 📜 Citations & Reflection

- **Citations**: [CITATIONS.md](file:///c:/Users/mskan/Desktop/Technical_Assyment_Cdazzdev/CITATIONS.md) documents all AI tool assistance and external references.
- **Reflection**: [REFLECTION.md](file:///c:/Users/mskan/Desktop/Technical_Assyment_Cdazzdev/REFLECTION.md) provides a concise (<600 words) engineering analysis of key architectural trade-offs, solutions, and future scaling pathways.
