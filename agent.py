"""
agent.py — The core agent loop.
Parses <tool_call> XML from the LLM, executes tools, feeds results back.
Token-efficient: caches tool results, compresses before sending to LLM.
Now supports MULTIPLE tool calls per LLM response (parallel execution).
"""

import json
import re
import hashlib
import plotly.graph_objects as go
from llm_client import call_llm
from tools import TOOLS, TOOLS_DESCRIPTION

# ── Tool result cache (survives for the session) ──────────────────────────────
_tool_cache: dict[str, str] = {}

def _cache_key(tool_name: str, tool_args: dict) -> str:
    return hashlib.md5(
        f"{tool_name}{json.dumps(tool_args, sort_keys=True)}".encode()
    ).hexdigest()

def _compress_tool_result(result: dict) -> str:
    """Send only what the LLM needs — headline numbers FIRST, then top records."""
    parts = []
    
    # ═══ HEADLINE NUMBERS (most important — LLM reads these first) ═══
    if "total_L" in result:
        parts.append(f"★★★ TOTAL CONSUMPTION = {result['total_L']:,.2f} L ★★★")
    if "num_readings" in result:
        parts.append(f"num_readings = {result['num_readings']}")
    if "totals" in result:
        parts.append(f"TOTALS BY CUSTOMER = {json.dumps(result['totals'], default=str)}")
    if "count" in result:
        parts.append(f"ANOMALY COUNT = {result['count']}")
    if "insight" in result:
        parts.append(f"INSIGHT: {result['insight']}")
    if "total_cost_TND" in result:
        parts.append(f"★★★ TOTAL COST = {result['total_cost_TND']} TND ★★★")
    if "volume_L" in result:
        parts.append(f"VOLUME = {result['volume_L']:,.2f} L = {result.get('volume_m3', '?')} m³")
    if "tariff_type" in result:
        parts.append(f"TARIFF TYPE = {result['tariff_type']}")
    if "anomaly_cost_TND" in result:
        parts.append(f"ANOMALY WASTE COST = {result['anomaly_cost_TND']} TND")
    if "tier_breakdown" in result:
        parts.append(f"TIER BREAKDOWN = {json.dumps(result['tier_breakdown'], default=str)}")
    if "pearson_r_temperature" in result:
        parts.append(f"PEARSON R (temperature) = {result['pearson_r_temperature']}")
        parts.append(f"PEARSON R (rainfall) = {result['pearson_r_rainfall']}")
    if "results" in result:
        # ISO results — show status summary
        statuses = {}
        for r in result["results"]:
            s = r.get("status", "?")
            statuses[s] = statuses.get(s, 0) + 1
        parts.append(f"ISO RESULTS SUMMARY = {json.dumps(statuses)}")
        parts.append(f"ISO FULL RESULTS = {json.dumps(result['results'], default=str)}")
    if "patterns" in result:
        parts.append(f"PATTERNS = {json.dumps(result['patterns'], default=str)}")
    
    # ═══ TOP RECORDS (for detail, clearly labeled as sample) ═══
    if "records" in result:
        top = result["records"][:10]
        parts.append(f"\n--- SAMPLE RECORDS (first 10 of {len(result['records'])} total) ---")
        parts.append(f"SAMPLE = {json.dumps(top, default=str)}")
    elif "table" in result:
        lines = result["table"].split("\n")[:15]
        parts.append(f"\n--- TABLE PREVIEW ---")
        parts.append("\n".join(lines))
    
    return "\n".join(parts).strip()


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Today's date is 2026-05-09. You are WaterSec, an expert AI water monitoring analyst with deep knowledge
of IoT sensor data, water consumption patterns, and water management best practices.

CRITICAL FORMATTING RULES:
- NEVER use LaTeX, math notation, or any TeX syntax (no \\frac, \\text, \\times, $...$, \\[ \\])
- Write all calculations in plain text: use * for multiply, / for divide, = for equals
- Example: "2796.63 / 6 = 466.1 minutes" NOT "\\frac{2796.63}{6}"

CRITICAL AGENT BEHAVIOUR RULES:
- You are an AUTONOMOUS AGENT, NOT a chatbot. You execute tools yourself — the user never sees tool calls.
- NEVER output text like "I will wait", "please provide", "once I receive", or "I will proceed with".
  These phrases are FORBIDDEN. Instead, just call the tools silently.
- When a user asks for MULTIPLE things (e.g., "compare A and B, show cost, weather, ISO"),
  call ALL needed tools in ONE response using multiple <tool_call> blocks.
- You can output up to 5 <tool_call> blocks in a single response — they all execute in parallel.
- For comparing two customers: generate ONE chart with chart_type="compare", not two separate charts.
- After receiving ALL tool results, synthesize ONE comprehensive final answer covering everything the user asked.
- Do NOT say "I will now proceed with..." — just call the tools silently.

EXAMPLE OF CORRECT MULTI-TOOL RESPONSE:
User: "Compare customerA and customerB cost and ISO"
Your response must be ONLY:
<tool_call>
{"tool": "calculate_water_cost", "args": {"customer": "customerA", "start_date": "2025-10-06", "end_date": "2026-05-09"}}
</tool_call>
<tool_call>
{"tool": "calculate_water_cost", "args": {"customer": "customerB", "start_date": "2025-10-06", "end_date": "2026-05-09"}}
</tool_call>
<tool_call>
{"tool": "check_iso24512_compliance", "args": {"customer": "customerA"}}
</tool_call>
<tool_call>
{"tool": "check_iso24512_compliance", "args": {"customer": "customerB"}}
</tool_call>
Do NOT write any text before, between, or after the tool calls.

DATASET YOU HAVE ACCESS TO:
- gym (Apr 2024 – May 2026): 4 shower cabins, each with Hot + Cold sensor = 8 devices total
  Devices: Gym Cabin 1/2/3/4 - Hot, Gym Cabin 1/2/3/4 - Cold
- customerA (May 2025 – Nov 2025): Office toilet blocs — 4 aggregated bloc sensors (cold only)
  Devices: Office Bloc A1, A2, A3, A4
- customerB (Oct 2025 – May 2026): Sanitary bloc — 1 sensor covering WCs, sinks, wudu sink (cold only)
  Devices: Sanitary Bloc B1
- customerC (Mar 2024 – Nov 2025): Residential home — 7 individual sensors (most granular)
  Devices: Residential Flush 1/2, Residential Sink 1/2/3, Residential Tap 1/2

DATA SCHEMA — HOW SENSOR EVENTS WORK:
- data_time = when the sensor SENT the reading = the moment the user CLOSED the tap
- data_period = how many seconds the water was flowing during that event
- event_start = data_time minus data_period = when the user OPENED the tap (computed on load)
- consumption_L = raw millilitres converted to litres (already done)
- flow_rate_Lpm = consumption_L / (data_period / 60) = litres per minute of actual flow
- Each row is one complete usage event (open → close), NOT a continuous time-series sample
- For sequence/pattern analysis, reason from event_start (when user acted), not data_time (when packet arrived)

WATER DOMAIN KNOWLEDGE:
- All consumption values are in LITRES (already converted from raw ml). Always report in litres.
- flow_rate_Lpm = litres per minute
- Benchmarks: shower ≈ 60L | toilet flush ≈ 6L | daily domestic ≈ 100–150 L/person/day
- Commercial toilet bloc (customerA, customerB): typical usage 50-200 users/day, expect 500-5000 L/day
- A hot shower sensor reading higher than its paired cold sensor = normal (hot water used more)
- Anomaly threshold: >2.5× rolling average is worth flagging
- Peak gym usage: typically 7–9 AM and 5–7 PM on weekdays
- Flush → Sink sequence within 5 minutes = normal handwashing behaviour
CRITICAL DATA QUALITY NOTES:
- Office toilet blocs (customerA, customerB) typically consume 100–500 L/day total.
- Residential (customerC) typically 100–400 L/day.
- If a tool returns a total >100,000 L for a toilet bloc, there were corrupted sensor readings that have been filtered out. The tool result's total_L is the CORRECT filtered value.
- Report the total_L value from the tool result EXACTLY as given — do not question it or recalculate from sample rows.

TOOL USAGE RULES:
- NEVER pass "all" for any tool argument — simply OMIT the argument entirely to mean no filter
- device_label and sub_category must be OMITTED (not set to "all", "any", or "*") when you want all devices
- generate_chart does NOT have a group_by="device_label" option — to compare devices use chart_type="compare" instead
- For hot vs cold comparison always use: chart_type="compare", customer="gym", no device_label filter
- When asked for data, trends, or comparisons → ALWAYS call query_data first, no exceptions
- NEVER say "no data available" without having called a tool first
- "last month" means the calendar month before today (2026-05-09), so start_date="2026-04-01", end_date="2026-04-30"
- "last 30 days" means start_date="2026-04-09", end_date="2026-05-09"
- Always resolve relative dates like "last month", "this week", "past 30 days" into explicit YYYY-MM-DD before calling a tool
- When asked for a chart/plot/graph/visualise → use generate_chart tool
- When asked about patterns or behaviour sequences → use detect_patterns
- When asked about unusual/anomalous/weird readings → use detect_anomalies
- Never invent numbers — always get them from a tool call
- customerC data ends at 2025-11-24 — "past 3 months" for customerC means 2025-09-01 to 2025-11-24, NOT relative to today
- Always check the customer's date range from the DATASET section before resolving relative dates
- For include_anomaly_cost, pass true (not "true") — it's a boolean
- ALWAYS generate a chart when:
  * user asks for comparison between devices, customers, or time periods
  * user asks for trends over time
  * user asks for anomalies (generate a line chart highlighting the anomalous period)
  * user asks about patterns (generate a bar chart of pattern occurrences)
  * any query that returns time-series data — always follow up with generate_chart
- The standard flow for most questions is: call query_data OR detect_* FIRST, then ALWAYS call generate_chart to visualise the result
- Only skip the chart for simple one-number answers like "what is the total"
- When reading tool results, ALWAYS use the total_L / total_cost_TND / insight fields directly.
  NEVER calculate totals by summing the sample records — they are only a preview.

TO CALL A TOOL, output EXACTLY this format (one <tool_call> block per tool, multiple blocks allowed):
<tool_call>
{"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}
</tool_call>

After receiving the tool result, give a clear, insightful analysis in plain language.
Format numbers cleanly: use L for litres, round to 2 decimal places.
When the data reveals something interesting, explain WHY it matters in water management context.

""" + TOOLS_DESCRIPTION


def _find_all_tool_calls(text: str) -> list[tuple[str, str, dict]]:
    """
    Find ALL <tool_call> blocks in the LLM response.
    Returns list of (full_match_text, tool_name, tool_args).
    """
    matches = re.finditer(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        text,
        re.DOTALL,
    )
    calls = []
    for match in matches:
        try:
            call = json.loads(match.group(1))
            tool_name = call.get("tool")
            tool_args = call.get("args", {})
            # Coerce "true"/"false" strings to Python booleans
            for k, v in tool_args.items():
                if isinstance(v, str) and v.lower() in ("true", "false"):
                    tool_args[k] = v.lower() == "true"
            calls.append((match.group(0), tool_name, tool_args))
        except json.JSONDecodeError:
            pass  # skip malformed calls
    return calls


def run_agent(user_message: str, history: list) -> tuple:
    """
    Main agent loop. Handles multi-turn conversation with tool calls.
    Supports MULTIPLE parallel tool calls per LLM response.

    Args:
        user_message: the user's latest message
        history: list of (user_msg, assistant_msg) tuples from Gradio

    Returns:
        (text_response, plotly_figure_or_None, provider_name)
    """
    # Build message history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_turn, assistant_turn in history:
        messages.append({"role": "user",      "content": user_turn})
        messages.append({"role": "assistant", "content": assistant_turn})
    messages.append({"role": "user", "content": user_message})

    chart    = None
    provider = "?"

    # Allow up to 4 iterations per turn (for multi-step chains like query→chart→cost→iso)
    for iteration in range(4):
        response_text, provider = call_llm(messages, max_tokens=2500)

        # Find ALL tool calls in this response
        tool_calls = _find_all_tool_calls(response_text)

        if not tool_calls:
            return response_text.strip(), chart, provider

        # ── Execute ALL tool calls in parallel ────────────────────────────────
        tool_results = []
        for full_tag, tool_name, tool_args in tool_calls:

            if tool_name not in TOOLS:
                tool_results.append(f"<tool_error>\nUnknown tool: {tool_name}\n</tool_error>")
                continue

            try:
                # ── Cache check ───────────────────────────────────────────────
                ck = _cache_key(tool_name, tool_args)
                if ck in _tool_cache and tool_name != "generate_chart":
                    print(f"[cache] HIT — {tool_name} {tool_args}")
                    tool_result = _tool_cache[ck]
                else:
                    result = TOOLS[tool_name](**tool_args)
                    print(f"[DEBUG] tool={tool_name} args={tool_args} "
                          f"total_L={result.get('total_L') if isinstance(result, dict) else 'figure'} "
                          f"records={len(result.get('records', [])) if isinstance(result, dict) else 'N/A'}")

                    # ── Plotly figure ─────────────────────────────────────────
                    if isinstance(result, go.Figure):
                        chart = result
                        tool_result = (
                            f"Chart generated successfully. "
                            f"Title: '{result.layout.title.text}'. "
                            f"The chart is displayed to the user."
                        )

                    # ── Empty result ──────────────────────────────────────────
                    elif isinstance(result, dict) and not result.get("records") and not result.get("table") and not result.get("total_cost_TND") and not result.get("results"):
                        tool_result = (
                            f"TOOL '{tool_name}' RETURNED NO DATA. This means your filter arguments matched zero rows.\n"
                            f"You called: {json.dumps({'tool': tool_name, 'args': tool_args}, default=str)}\n"
                            f"COMMON CAUSES:\n"
                            f"- You passed a string like 'all' for device_label or sub_category — NEVER do this, omit the argument entirely\n"
                            f"- Your date range is wrong or out of the customer's data period\n"
                            f"- Check the LIVE DATASET SUMMARY for valid date ranges\n"
                            f"RETRY the tool call with corrected arguments."
                        )

                    # ── Normal result — compress before sending ───────────────
                    else:
                        tool_result = _compress_tool_result(result)
                        # Cache it for future identical calls
                        _tool_cache[ck] = tool_result

                tool_results.append(
                    f"<tool_result tool=\"{tool_name}\">\n{tool_result}\n</tool_result>"
                )

            except Exception as e:
                tool_results.append(
                    f"<tool_error tool=\"{tool_name}\">\nTool execution error: {e}\n</tool_error>"
                )

        # ── Feed ALL results back at once ─────────────────────────────────────
        combined_results = "\n\n".join(tool_results)

        messages.append({"role": "assistant", "content": response_text})
        messages.append({
            "role": "user",
            "content": (
                f"ALL TOOL RESULTS (all {len(tool_calls)} tool calls executed successfully):\n\n"
                f"{combined_results}\n\n"
                f"---\n"
                f"IMPORTANT INSTRUCTIONS:\n"
                f"- All {len(tool_calls)} tools executed successfully. The results are above.\n"
                f"- Do NOT say there is 'no data' unless a tool explicitly returned empty results.\n"
                f"- Do NOT re-call any of the same tools with the same arguments — results are final.\n"
                f"- If you called query_data or detect_anomalies and haven't called generate_chart yet, call it now.\n"
                f"- If the user asked for cost, weather, or ISO compliance and you haven't called those tools yet, call them now.\n"
                f"- If ALL user requests are satisfied, provide your comprehensive final analysis.\n"
                f"- Synthesize ALL results into ONE final answer covering everything the user asked.\n"
            ),
        })

    return response_text.strip(), chart, provider