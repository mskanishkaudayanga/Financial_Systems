# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create task3_financial_agents/run_demo.py to allow running the single autonomous agent demo from terminal with real-time trace logging', Date: 2026-09-11
"""
Standalone Demonstration Script for Task 3A Autonomous Financial Research Agent.

Run this script to watch the LangGraph agent autonomously inspect state,
decide tool calls, observe results, and generate the final report.

Usage:
    python run_demo.py [TICKER]
Example:
    python run_demo.py AAPL
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.workflows import run_financial_research_agent


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"

    print("=" * 85)
    print(f"🚀 RUNNING AUTONOMOUS FINANCIAL RESEARCH AGENT FOR TICKER: {ticker}")
    print("=" * 85 + "\n")

    try:
        final_state = run_financial_research_agent(ticker=ticker)

        print("\n" + "=" * 85)
        print("📄 FINAL SYNTHESIZED EQUITY RESEARCH REPORT")
        print("=" * 85 + "\n")
        print(final_state.get("final_report", "No report generated."))
        print("\n" + "=" * 85)
        print("✅ RESEARCH WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 85)

    except Exception as exc:
        print(f"\n❌ Execution Error: {exc}")
        print("\nNote: Make sure your LLM_API_KEY is configured in .env for full live execution.")


if __name__ == "__main__":
    main()
