# AI-ASSISTED: Gemini (gemini-3.6-flash), Prompt: 'Update tracer.py adding UPDATED OBSERVATION event formatting with datetime.now(timezone.utc)', Date: 2026-09-11
"""
Observability and Trace Logging Infrastructure.

Emits structured execution logs (AGENT, TOOL CALL, TOOL RESULT, UPDATED OBSERVATION, AGENT DECISION)
to standard output (notebook/terminal) and persistent agent_trace.jsonl file.
"""

import json
from datetime import datetime, timezone
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
    - TOOL RESULT: Raw output returned by executed tool
    - UPDATED OBSERVATION: State observation summary recorded from tool output
    - AGENT DECISION: Strategic choice, replan, or state transition decision
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    meta = metadata or {}

    header_box = f"[{event_type}] ({timestamp[:19]})"

    if event_type == "AGENT":
        print(f"\n🤖 {header_box}")
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
        print(f"   Result Summary: {content[:250]}..." if len(content) > 250 else f"   Result Summary: {content}")

    elif event_type == "UPDATED OBSERVATION":
        tool_name = meta.get("tool_name", "UnknownTool")
        print(f"\n👁️  {header_box}")
        print(f"   Observation ({tool_name}): {content}")

    elif event_type == "AGENT DECISION":
        print(f"\n💡 {header_box}")
        print(f"   Decision: {content}")

    else:
        print(f"\nℹ️  {header_box} {content}")

    # Append event to agent_trace.jsonl
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
        pass
