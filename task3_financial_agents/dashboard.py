# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create read-only Streamlit observability dashboard reading agent_trace.jsonl', Date: 2026-09-11
"""
Streamlit Observability Dashboard for Financial AI Agent System (Task 3 Bonus)

Displays execution metrics, tool usage breakdowns, and chronological agent execution traces
from `agent_trace.jsonl` in a read-only view.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="Financial Agent Observability Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stCodeBlock {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


def load_trace_data(trace_path: Path) -> list:
    """Read trace entries from agent_trace.jsonl safely (Read-Only)."""
    if not trace_path.exists():
        st.warning(f"Trace file not found at: {trace_path}")
        return []

    records = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data["_line_number"] = line_num
                records.append(data)
            except json.JSONDecodeError:
                continue
    return records


def calculate_metrics(records: list):
    """Calculate summary metrics from trace records."""
    tool_calls = [r for r in records if r.get("event_type") == "TOOL CALL"]
    tool_results = [r for r in records if r.get("event_type") == "TOOL RESULT"]
    decisions = [r for r in records if r.get("event_type") == "AGENT DECISION"]
    handoffs = [r for r in records if r.get("event_type") == "HANDOFF"]

    # Calculate execution duration from first to last valid ISO timestamp
    timestamps = []
    for r in records:
        ts_str = r.get("timestamp")
        if ts_str:
            try:
                # Standardize ISO timestamps
                ts_clean = ts_str.replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts_clean)
                timestamps.append(dt)
            except ValueError:
                pass

    if timestamps and len(timestamps) > 1:
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        duration_sec = (max_ts - min_ts).total_seconds()
        duration_str = f"{duration_sec:.2f} seconds" if duration_sec < 120 else f"{duration_sec / 60:.2f} minutes"
    else:
        duration_str = "N/A"

    # Tool call breakdown by tool name
    tool_counts = {}
    for tc in tool_calls:
        tool_name = tc.get("metadata", {}).get("tool_name", "unknown")
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

    return {
        "total_events": len(records),
        "total_tool_calls": len(tool_calls),
        "total_tool_results": len(tool_results),
        "total_decisions": len(decisions),
        "total_handoffs": len(handoffs),
        "duration_str": duration_str,
        "tool_counts": tool_counts
    }


def main():
    st.title("📊 Financial Agent Observability Dashboard")
    st.markdown("Read-only audit trace visualizer for **`agent_trace.jsonl`**.")

    # Sidebar File Selection & Filters
    st.sidebar.header("⚙️ Configuration & Filters")

    default_trace_path = Path(__file__).parent / "agent_trace.jsonl"
    trace_path_input = st.sidebar.text_input("Trace File Path", str(default_trace_path))
    trace_path = Path(trace_path_input)

    st.sidebar.divider()

    records = load_trace_data(trace_path)

    if not records:
        st.info("No trace records available. Execute a research workflow or demo script to generate trace data.")
        return

    metrics = calculate_metrics(records)

    # Top Metric Bar
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Events", metrics["total_events"])
    col2.metric("Total Tool Calls", metrics["total_tool_calls"])
    col3.metric("Tool Results", metrics["total_tool_results"])
    col4.metric("Agent Decisions", metrics["total_decisions"])
    col5.metric("Trace Duration", metrics["duration_str"])

    st.divider()

    # Tool Usage Breakdown Section
    st.subheader("🛠️ Tool Invocations by Tool Name")
    if metrics["tool_counts"]:
        df_tools = pd.DataFrame(
            list(metrics["tool_counts"].items()),
            columns=["Tool Name", "Invocation Count"]
        ).sort_values(by="Invocation Count", ascending=False)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.dataframe(df_tools, hide_index=True, use_container_width=True)

        with col_right:
            st.bar_chart(df_tools.set_index("Tool Name"))
    else:
        st.info("No tool call events found in trace file.")

    st.divider()

    # Chronological Execution Trace Section
    st.subheader("⏱️ Chronological Execution Trace")

    # Filter Options
    event_types = sorted(list(set(r.get("event_type", "UNKNOWN") for r in records if "event_type" in r)))
    selected_events = st.sidebar.multiselect("Filter by Event Type", options=event_types, default=event_types)

    tool_names = list(metrics["tool_counts"].keys())
    selected_tools = st.sidebar.multiselect("Filter by Tool Name", options=tool_names, default=tool_names)

    # Filter records
    filtered_records = []
    for r in records:
        e_type = r.get("event_type")
        if e_type and e_type not in selected_events:
            continue
        t_name = r.get("metadata", {}).get("tool_name")
        if t_name and selected_tools and t_name not in selected_tools:
            continue
        filtered_records.append(r)

    st.caption(f"Showing {len(filtered_records)} of {len(records)} trace events.")

    # Chronological Table / Details View
    for idx, r in enumerate(filtered_records, 1):
        ts = r.get("timestamp", "N/A")
        e_type = r.get("event_type", r.get("event", "EVENT"))
        meta = r.get("metadata", {})
        tool_name = meta.get("tool_name", "")
        content = r.get("content", "")

        title_str = f"#{idx} | [{ts}] {e_type}"
        if tool_name:
            title_str += f" -> {tool_name}"

        with st.expander(title_str, expanded=False):
            c_left, c_right = st.columns([1, 2])

            with c_left:
                st.markdown("**Metadata & Inputs**")
                st.json({
                    "timestamp": ts,
                    "event_type": e_type,
                    "tool_name": tool_name,
                    "arguments": meta.get("args", "N/A"),
                    "status": meta.get("status", "N/A"),
                    "call_id": meta.get("call_id", "N/A"),
                    "line_number": r.get("_line_number")
                })

            with c_right:
                st.markdown("**Output / Content Preview**")
                if content:
                    if isinstance(content, str):
                        try:
                            # Parse JSON string if content is JSON
                            parsed = json.loads(content)
                            st.json(parsed)
                        except Exception:
                            st.code(content, language="markdown")
                    else:
                        st.json(content)
                else:
                    st.text("No content body.")


if __name__ == "__main__":
    main()
