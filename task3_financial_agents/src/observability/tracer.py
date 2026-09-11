# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Create src/observability/tracer.py formatting AGENT, TOOL CALL, TOOL RESULT, AGENT DECISION steps to stdout and agent_trace.jsonl', Date: 2026-09-11
"""
Observability and Trace Logging Infrastructure.

Emits structured execution logs (AGENT, TOOL CALL, TOOL RESULT, AGENT DECISION)
to both standard output (notebook/terminal) and persistent agent_trace.jsonl file.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional
from src.config import config


def log_trace_event(
    event_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an agent trace step with rich console formatting and JSONL persistence.

    Supported event types:
    - AGENT: Reasoning / system evaluation by the agent
    - TOOL CALL: Tool invocation details (tool name, arguments)
    - TOOL RESULT: Output returned by executed tool
    - AGENT DECISION: Strategic choice or state transition decision

    Args:
        event_type: One of 'AGENT', 'TOOL CALL', 'TOOL RESULT', 'AGENT DECISION'.
        content: Main text summary or log body.
        metadata: Additional structured key-value metadata.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    meta = metadata or {}

    # 1. Console / Notebook trace formatting
    header_box = f"[{event_type}] ({timestamp[:19]})"

    if event_type == "AGENT":
        print(f"\n🤖 \031{header_box}\031")
        print(f"   Reasoning: {content}")

    elif event_type == "TOOL CALL":
        tool_name = meta.get("tool_name", "UnknownTool")
        args_str = json.dumps(meta.get("args", {}))
        print(f"\n🛠️  {header_box}")
        print(f"   Tool: {tool_name}")
        print(f"   Arguments: {args_str}")

    elif event_type == "TOOL RESULT":
        tool_name = meta.get("tool_name", "UnknownTool")
        status = meta.get("status", "success")
        print(f"\n📊 {header_box}")
        print(f"   Tool: {tool_name} | Status: {status}")
        print(f"   Result Summary: {content[:300]}..." if len(content) > 300 else f"   Result Summary: {content}")

    elif event_type == "AGENT DECISION":
        print(f"\n💡 {header_box}")
        print(f"   Decision: {content}")

    else:
        print(f"\nℹ️  {header_box} {content}")

    # 2. Append event to agent_trace.jsonl
    try:
        config.TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "content": content,
            "metadata": meta,
        }
        with open(config.TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass  # Defensive non-blocking logging
