"""
benchmark_models_report.py — WaterSec Model Benchmarking & Cost Analysis Tool
==============================================================================
Compares cloud (Groq) and local (Ollama) models on water domain prompts.
Generates a PDF report with responses, scores, and cost analysis.

v5 changes:
  - Fixed auto-score judge: guards against [NO_RESPONSE] sentinel, increased max_tokens to 100
  - Fixed Ollama timeout crash: explicit timeout=120, broad except in run_prompt
  - Fixed judge model ID: uses "openai/gpt-oss-120b" with fallback to llama-3.3-70b
  - Token-safe: single attempt per call, no retry loops

Usage:
    python benchmark_models_report.py
"""

import time
import os
import sys
import json
import math
import datetime
import subprocess
import requests
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — AUTO-START OLLAMA (same logic as app.py)
# ═══════════════════════════════════════════════════════════════════════════════
def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        print("[ollama] Already running.")
        return
    except Exception:
        pass
    print("[ollama] Starting Ollama in background...")
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(["ollama", "serve"], **kwargs)
    for _ in range(15):
        time.sleep(1)
        try:
            requests.get("http://localhost:11434/api/tags", timeout=2)
            print("[ollama] Ready.")
            return
        except Exception:
            pass
    print("[ollama] WARNING: Ollama did not start in time — local models may fail.")


def ensure_model_pulled(model_id: str):
    """Pull an Ollama model if not already downloaded."""
    try:
        resp  = requests.get("http://localhost:11434/api/tags", timeout=3)
        names = [m["name"] for m in resp.json().get("models", [])]
        base  = model_id.split(":")[0]
        if any(base in n for n in names):
            print(f"[ollama] '{model_id}' already available.")
            return
    except Exception:
        pass
    print(f"[ollama] Pulling '{model_id}' — may take a few minutes on first run...")
    try:
        subprocess.run(["ollama", "pull", model_id], check=True)
        print(f"[ollama] '{model_id}' ready.")
    except Exception as e:
        print(f"[ollama] WARNING: Could not pull '{model_id}': {e}")


ensure_ollama_running()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD REAL WATERSEC DATA
# ═══════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_DATA_LOADED = False
_DF          = None

try:
    import tools as _tools
    from data_loader import load_and_clean as _load_clean
    _DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    _tools.DF    = _load_clean(_DATA_DIR)
    _DF          = _tools.DF
    _DATA_LOADED = True
    print(f"[data] {len(_DF):,} rows loaded — real data will be injected into prompts.\n")
except Exception as _err:
    print(f"[data] WARNING: Could not load CSVs ({_err}).\n"
          f"       Prompts will use pre-computed fallback values.\n")


def _live(fn, fallback: str) -> str:
    """Run fn() if data loaded, else return fallback string."""
    if not _DATA_LOADED:
        return fallback
    try:
        return fn()
    except Exception as e:
        return f"{fallback}  [live fetch failed: {e}]"


def _make_prompt(question: str, data_fn, fallback: str) -> str:
    data_block = _live(data_fn, fallback)
    sep = "\n\nDATA FROM WATERSEC SENSORS:\n"
    return question + (sep + data_block if data_block.strip() else "")


# ── Live data builders ────────────────────────────────────────────────────────
def _gym_monthly():
    r = _tools.query_data(customer="gym", group_by="month", metric="sum")
    return r.get("table", "").replace("nan", "N/A")

def _hot_cold():
    rows = []
    for cabin in ["1", "2", "3", "4"]:
        h  = _tools.query_data(device_label=f"Cabin {cabin} - Hot",  metric="sum")
        c  = _tools.query_data(device_label=f"Cabin {cabin} - Cold", metric="sum")
        ht = h.get("total_L", 0)
        ct = c.get("total_L", 0)
        rows.append(f"Cabin {cabin} — Hot: {ht:>10,.1f} L | Cold: {ct:>9,.1f} L | Ratio: {ht/(ct or 1):.1f}x")
    return "\n".join(rows)

def _anomalies_c():
    r = _tools.detect_anomalies(customer="customerC", window_days=7)
    return r.get("table", "")[:900]

def _daily_avg():
    rows = []
    for cust, grp in _DF.groupby("customer"):
        days  = max((grp["data_time"].max() - grp["data_time"].min()).days, 1)
        total = grp["consumption_L"].sum()
        rows.append(f"  {cust:<12}: {total/days:>8.1f} L/day  |  total {total:>10,.1f} L over {days} days")
    return "\n".join(rows)

def _a_vs_b():
    lines = []
    for c in ["customerA", "customerB"]:
        g = _DF[_DF["customer"] == c]
        lines.append(
            f"  {c}: {g['consumption_L'].sum():>10,.1f} L | "
            f"{g['data_time'].min().date()} → {g['data_time'].max().date()} | "
            f"sensors: {g['device_label'].nunique()}"
        )
    return "\n".join(lines)

def _patterns():
    r = _tools.detect_patterns(window_minutes=5)
    return r.get("table", "") + "\n" + r.get("insight", "")

def _gym_hourly():
    hot = _DF[(_DF["customer"] == "gym") & (_DF["device_label"].str.contains("Hot"))].copy()
    hot["hour"] = hot["data_time"].dt.hour
    peak = hot.groupby("hour")["consumption_L"].sum().nlargest(5)
    lines = ["Top 5 peak hours (hot water, total litres all time):"]
    for h, v in peak.items():
        lines.append(f"  {h:02d}:00 — {v:,.1f} L")
    lines.append(f"Flow rate (5th–95th percentile): "
                 f"{hot['flow_rate_Lpm'].quantile(0.05):.2f} – "
                 f"{hot['flow_rate_Lpm'].quantile(0.95):.2f} L/min")
    lines.append(f"Median flow rate: {hot['flow_rate_Lpm'].median():.2f} L/min")
    return "\n".join(lines)

def _session_stats():
    r = _tools.session_analysis()
    return r.get("table", "")[:900]


# ═══════════════════════════════════════════════════════════════════════════════
# GROQ SMART ROUTER — KEY_1 → KEY_N with per-key cooldown
# ═══════════════════════════════════════════════════════════════════════════════
_raw_keys = []
for _i in range(1, 100):
    _k = os.environ.get(f"GROQ_API_KEY_{_i}", "")
    if not _k:
        break
    _raw_keys.append(_k)
_groq_keys    = [k for k in _raw_keys if k and not k.startswith("your_")]
_groq_clients = [OpenAI(base_url="https://api.groq.com/openai/v1", api_key=k) for k in _groq_keys]
_groq_cooldowns: dict[int, float] = {}
_ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=120)

print(f"[benchmark] Groq keys loaded: {len(_groq_keys)} "
      + ("(" + ", ".join(f"KEY_{i+1}" for i in range(len(_groq_keys))) + ")"
         if _groq_keys else "(none — only local models available)"))


def _call_groq(model_id, messages, max_tokens, temperature):
    """Single attempt across available Groq keys. No retry loop — one shot per key."""
    now       = time.time()
    available = [i for i in range(len(_groq_clients)) if _groq_cooldowns.get(i, 0) <= now]
    if not available:
        soonest = min(range(len(_groq_clients)), key=lambda i: _groq_cooldowns.get(i, 0))
        wait    = max(0, _groq_cooldowns[soonest] - now)
        if 0 < wait <= 90:
            print(f"[benchmark] All keys cooling — waiting {wait:.0f}s...")
            time.sleep(wait + 1)
            available = [soonest]
        else:
            raise RuntimeError("All Groq keys exhausted.")
    for idx in available:
        lbl = f"KEY_{idx+1}"
        try:
            start   = time.time()
            resp    = _groq_clients[idx].chat.completions.create(
                model=model_id, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            elapsed = round(time.time() - start, 2)
            content = resp.choices[0].message.content
            if not content or not content.strip():
                reason = getattr(resp.choices[0], "finish_reason", "unknown")
                print(f"[benchmark] ! Groq {lbl} empty content "
                      f"(finish_reason={reason}, {elapsed}s) -> NO_RESPONSE")
                return "[NO_RESPONSE]", elapsed
            print(f"[benchmark] Groq {lbl} -> {elapsed}s")
            return content.strip(), elapsed
        except RateLimitError:
            print(f"[benchmark] Groq {lbl} rate-limited -- 60s cooldown")
            _groq_cooldowns[idx] = time.time() + 60
        except APIError as e:
            print(f"[benchmark] Groq {lbl} API error: {e}")
        except Exception as e:
            print(f"[benchmark] Groq {lbl} unexpected: {e}")
    raise RuntimeError("All available Groq keys failed.")


def _call_ollama(model_id, messages, max_tokens, temperature):
    """Single attempt at Ollama. Catches all exceptions including timeouts."""
    start = time.time()
    try:
        resp    = _ollama_client.chat.completions.create(
            model=model_id, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        elapsed = round(time.time() - start, 2)
        content = resp.choices[0].message.content
        if not content or not content.strip():
            print(f"[ollama] {model_id} empty response -> NO_RESPONSE")
            return "[NO_RESPONSE]", elapsed
        print(f"[ollama] {model_id} -> {elapsed:.1f}s")
        return content.strip(), elapsed
    except Exception as e:
        print(f"[ollama] {model_id} ERROR: {type(e).__name__}: {e}")
        return f"ERROR: {type(e).__name__}: {e}", round(time.time()-start, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
REGISTRY = [
    ("groq",   "llama-3.1-8b-instant",   "Llama 3.1 8B      (Groq)",  0.05,  0.08,  560,  0),
    ("groq", "meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B  (Groq)", 0.11, 0.34, 750, 0),
    ("groq",   "qwen/qwen3-32b",          "Qwen3 32B         (Groq)",  0.29,  0.59,  535,  0),
    ("groq",   "llama-3.3-70b-versatile", "Llama 3.3 70B     (Groq)",  0.59,  0.79,  280,  0),
    ("ollama", "mistral:7b-instruct",     "Mistral 7B        (Local)", 0.0,   0.0,   None, 83),
    ("ollama", "qwen2.5:3b",              "Qwen2.5 3B        (Local)", 0.0,   0.0,   None, 83),
]

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS — built at runtime so live data is always fresh
# ═══════════════════════════════════════════════════════════════════════════════
def _build_prompts() -> list[dict]:
    return [
        {"id": "S1", "category": "Standard", "difficulty": "Easy",
         "prompt": _make_prompt(
             "What is the total water consumption at the gym last month? "
             "The gym has 4 shower cabins, each with a hot and cold sensor (8 sensors total). "
             "Analyse the monthly trend and state the most recent full month total in litres.",
             _gym_monthly,
             "| period | sum_L |\n| 2026-03-31 | 7825.56 |\n| 2026-04-30 | 10898.77 |",
         )},
        {"id": "S2", "category": "Standard", "difficulty": "Easy",
         "prompt": _make_prompt(
             "Compare hot vs cold water usage across all 4 gym shower cabins. "
             "Calculate the hot-to-cold ratio per cabin and identify which cabin "
             "uses the most hot water overall. What does this suggest operationally?",
             _hot_cold,
             "Cabin 1 Hot: 45386 L Cold: 10664 L\nCabin 2 Hot: 39223 L Cold: 7040 L",
         )},
        {"id": "S3", "category": "Standard", "difficulty": "Medium",
         "prompt": _make_prompt(
             "Identify and explain the top anomalies in the residential CustomerC dataset. "
             "CustomerC has 7 sensors: Flush 1/2, Sink 1/2/3, Tap 1/2. "
             "A normal flush is ~6 L and a normal sink use is ~1-2 L. "
             "For each anomaly listed, explain the likely cause and recommended action.",
             _anomalies_c,
             "Residential Flush 1: 149.61 L (z=14.3) on 2024-06-25",
         )},
        {"id": "S4", "category": "Standard", "difficulty": "Medium",
         "prompt": _make_prompt(
             "Which customer has the highest average daily water consumption? "
             "Rank all 4 customers and explain the difference given their profiles: "
             "Gym (showers), CustomerA (office toilet bloc), "
             "CustomerB (sanitary bloc with wudu sink), CustomerC (residential home).",
             _daily_avg,
             "customerA: 604 L/day | customerB: 1003 L/day | customerC: 258 L/day | gym: 218 L/day",
         )},
        {"id": "S5", "category": "Standard", "difficulty": "Medium",
         "prompt": _make_prompt(
             "Compare CustomerA and CustomerB water consumption side by side. "
             "CustomerA is an office toilet bloc with 4 aggregated bloc sensors. "
             "CustomerB is a sanitary bloc (2 WCs, sinks, wudu sink) with 1 aggregated sensor. "
             "Explain the difference in total and daily consumption. Which is more water-efficient per user?",
             _a_vs_b,
             "customerA: 105712 L (May–Nov 2025)\ncustomerB: 214731 L (Oct 2025–May 2026)",
         )},
        {"id": "A1", "category": "Advanced", "difficulty": "Hard",
         "prompt": _make_prompt(
             "Analyse the behavioural usage sequences detected in the residential CustomerC dataset "
             "within 5-minute windows. What do these sequences reveal about occupant behaviour "
             "and bathroom infrastructure? Provide specific water conservation recommendations "
             "backed by the occurrence numbers in the data.",
             _patterns,
             "Sink→Sink: 6,222,161 | Sink→Flush: 4,515,972 | Flush→Sink: 4,403,638",
         )},
        {"id": "A2", "category": "Advanced", "difficulty": "Hard",
         "prompt": _make_prompt(
             "Design a demand-based hot water heating schedule for the gym based on the real "
             "hourly sensor data. Identify peak hours, off-peak windows, and recommend specific "
             "pre-heating start times. Estimate expected energy savings as a percentage. "
             "Account for the fact that hot water usage is ~4x higher than cold water usage.",
             _gym_hourly,
             "Peak hours: 20:00-22:00 (18362 L), 19:00 (9466 L), 17:00 (9195 L)",
         )},
        {"id": "A3", "category": "Advanced", "difficulty": "Hard",
         "prompt": _make_prompt(
             "CustomerB has a single aggregated sensor covering 2 WCs (man/woman), each with "
             "a central sink, 2 flushes, 2 flexibles, and 1 wudu sink — 9+ fixtures on 1 sensor. "
             "What are the specific risks of this architecture for leak detection and anomaly isolation? "
             "Redesign the sensor layout: state the minimum number of sensors needed and where to place them.",
             _a_vs_b,
             "customerB: 214731 L total, 1 sensor, Oct 2025–May 2026",
         )},
        {"id": "A4", "category": "Advanced", "difficulty": "Expert",
         "prompt": _make_prompt(
             "Rank the 4 monitored sites by leak detection risk (highest to lowest): "
             "gym showers (8 sensors, hot+cold), office toilets (4 aggregated bloc sensors), "
             "sanitary bloc (1 aggregated sensor), residential (7 granular sensors). "
             "For each site: (1) justify the ranking, (2) set an anomaly threshold in litres per event, "
             "(3) define the full alert action chain (who is notified, in what timeframe, via what channel).",
             _session_stats,
             "Avg gym session: 6.11 L | avg customerC session: 2.19 L",
         )},
        {"id": "A5", "category": "Advanced", "difficulty": "Expert",
         "prompt": (
             "Residential Flush 2 recorded 2796.63 L in a single event (z-score: 11.75). "
             "A normal flush is 6 L. Flow rate during this event was 6 L/min.\n\n"
             "Calculate step by step:\n"
             "  1. Exact duration of the event in minutes and hours\n"
             "  2. Water cost at 0.003 EUR/L\n"
             "  3. CO2 equivalent at 0.298 kg CO2 per m³ of water\n\n"
             "Then rank the three most likely root causes by probability, "
             "explain each, and state what WaterSec alert should have fired."
         )},
        {"id": "A6", "category": "Advanced", "difficulty": "Expert",
         "prompt": _make_prompt(
             "A water utility wants to use WaterSec to meet ISO 24512 water service management standards. "
             "Based on the 4 customer profiles and their sensor granularity, identify: "
             "(1) which ISO 24512 KPIs can be computed directly from this sensor data and how, "
             "(2) which KPIs require additional data sources — name those sources, "
             "(3) propose a monthly compliance report structure with specific named sections and metrics.",
             _daily_avg,
             "4 customers: gym (8 sensors), customerA (4 sensors), customerB (1 sensor), customerC (7 sensors)",
         )},
        {"id": "T1", "category": "Tool-Use", "difficulty": "Easy",
         "prompt": (
             "Compare hot vs cold water usage across all gym shower cabins. "
             "To answer this you must:\n"
             "1. Call query_data with customer='gym', sub_category='Hot' to get hot totals\n"
             "2. Call query_data with customer='gym', sub_category='Cold' to get cold totals\n"
             "3. Call generate_chart with chart_type='compare', customer='gym' to visualise\n"
             "State the tool calls you would make and what arguments you would use."
         )},
        {"id": "T2", "category": "Tool-Use", "difficulty": "Medium",
         "prompt": (
             "Plot the daily consumption trend for customerC over the past 3 months. "
             "To answer this you must:\n"
             "1. Call query_data with customer='customerC', group_by='day', "
             "start_date='2025-09-01', end_date='2025-11-24'\n"
             "2. Call generate_chart with chart_type='line', customer='customerC', "
             "group_by='day', start_date='2025-09-01', end_date='2025-11-24'\n"
             "State the exact tool calls and arguments you would use. "
             "Note: customerC data ends at 2025-11-24, NOT relative to today."
         )},
        {"id": "T3", "category": "Tool-Use", "difficulty": "Hard",
         "prompt": (
             "Detect anomalies for customerC and then visualise the result as a line chart. "
             "To answer this you must:\n"
             "1. Call detect_anomalies with customer='customerC', window_days=7\n"
             "2. Call generate_chart with chart_type='line', customer='customerC' "
             "to show the consumption trend with context\n"
             "IMPORTANT: Never pass device_label='all' — omit it entirely to include all devices. "
             "State the exact tool calls and arguments. "
             "Flag any argument that would be WRONG to pass."
         )},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════
WEIGHTS               = {"relevance": 0.25, "correctness": 0.30, "insight": 0.25, "domain": 0.20}
HALLUCINATION_PENALTY = 0.5

SYSTEM_PROMPT = """You are WaterSec, an expert AI water monitoring analyst.
CRITICAL: You do NOT have access to any tools or functions. Never call tools. When asked what tool calls you would make, describe them in plain text only.
CRITICAL FORMATTING RULES:
- NEVER use LaTeX, math notation, or any TeX syntax (no \\frac, \\text, \\times, $...$, \\[ \\])
- Write all calculations in plain text: use * for multiply, / for divide, = for equals
- Example: "2796.63 / 6 = 466.1 minutes" NOT "\\frac{2796.63}{6} = 466.1"
Each data row = one complete usage event (tap open → close → data sent).
consumption_L = total litres used in that event.
session_duration_s = how long the water flowed in seconds.
Always base your analysis on the DATA provided in the prompt.
Use litres (L) and L/min as units. Be specific with numbers.
Do not invent values not present in the provided data."""

MONTHLY_QUERIES   = 50 * 30
AVG_INPUT_TOKENS  = 900
AVG_OUTPUT_TOKENS = 700


def compute_monthly_cost(inp, out, provider):
    if provider == "ollama":
        return 0.0
    return round((MONTHLY_QUERIES * AVG_INPUT_TOKENS / 1_000_000) * inp +
                 (MONTHLY_QUERIES * AVG_OUTPUT_TOKENS / 1_000_000) * out, 4)


def compute_score(scores: dict) -> float:
    base    = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    penalty = HALLUCINATION_PENALTY * scores.get("hallucination", 0)
    return round(max(0, base - penalty), 3)


def run_prompt(model_entry: tuple, prompt: str, max_tokens: int = 900) -> tuple[str, float]:
    """Single attempt — no retry loop. Returns (response_text, elapsed_seconds)."""
    provider, model_id, label, *_ = model_entry
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    if provider == "groq":
        try:
            return _call_groq(model_id, messages, max_tokens, 0.2)
        except RuntimeError as e:
            return f"ERROR (all Groq keys exhausted): {e}", 0.0
    else:
        try:
            return _call_ollama(model_id, messages, max_tokens, 0.2)
        except Exception as e:
            return f"ERROR (Ollama crash): {type(e).__name__}: {e}", 0.0


def get_score(criterion: str, scale: str) -> float:
    while True:
        try:
            val = float(input(f"    {criterion} {scale}: ").strip())
            lo, hi = (0, 2) if "hallucination" in criterion.lower() else (1, 5)
            if lo <= val <= hi:
                return val
            print(f"    Enter {lo}–{hi}.")
        except ValueError:
            print("    Enter a number.")


def score_response(label: str) -> dict:
    print(f"\n  Scoring: {label}")
    print("  ─────────────────────────────")
    scores = {
        "relevance":     get_score("Relevance    ", "[1-5]"),
        "correctness":   get_score("Correctness  ", "[1-5]"),
        "insight":       get_score("Insight      ", "[1-5]"),
        "domain":        get_score("Domain aware ", "[1-5]"),
        "hallucination": get_score("Hallucination", "[0=none 1=minor 2=major]"),
    }
    scores["final"] = compute_score(scores)
    print(f"  → Final: {scores['final']:.3f} / 5.0")
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# PDF REPORT (reportlab — no LaTeX needed)
# ═══════════════════════════════════════════════════════════════════════════════
def _sanitize(text: str) -> str:
    """Strip unicode chars that Helvetica cannot render, and replace LaTeX remnants."""
    return (text
        .replace("\u202f", " ")   # narrow no-break space
        .replace("\u2011", "-")   # non-breaking hyphen
        .replace("\u2010", "-")   # hyphen
        .replace("\u2012", "-")   # figure dash
        .replace("\u2013", "-")   # en dash
        .replace("\u2014", "--")  # em dash
        .replace("\u00d7", "x")   # multiplication sign
        .replace("\u00f7", "/")   # division sign
        .replace("\u2019", "'")   # right single quote
        .replace("\u2018", "'")   # left single quote
        .replace("\u201c", '"')   # left double quote
        .replace("\u201d", '"')   # right double quote
        .replace("\\frac", "")
        .replace("\\text", "")
        .replace("\\times", "x")
        .replace("$", "")
    )


def generate_pdf(results: list, models: list, run_date: str, pdf_path: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether, ListFlowable, ListItem,
    )

    BLUE      = colors.HexColor("#0078D7")
    GREEN     = colors.HexColor("#228B22")
    RED       = colors.HexColor("#C80000")
    GOLD      = colors.HexColor("#B8860B")
    LIGHT     = colors.HexColor("#F0F6FF")
    DARK      = colors.HexColor("#1A1A2E")
    GREY      = colors.HexColor("#666666")
    WHITE     = colors.white
    W, H      = A4

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    title_s  = S("T",  fontSize=22, textColor=DARK,  alignment=TA_CENTER, spaceAfter=10, fontName="Helvetica-Bold")
    sub_s    = S("Su", fontSize=13, textColor=GREY,  alignment=TA_CENTER, spaceAfter=8,  fontName="Helvetica")
    date_s   = S("D",  fontSize=9,  textColor=GREY,  alignment=TA_CENTER, spaceAfter=20, fontName="Helvetica-Oblique")
    sec_s    = S("Se", fontSize=13, textColor=WHITE, spaceAfter=6, spaceBefore=14,
                       fontName="Helvetica-Bold", backColor=BLUE, leftIndent=-12, rightIndent=-12, leading=18)
    sub2_s   = S("S2", fontSize=11, textColor=DARK,  spaceAfter=4, spaceBefore=10, fontName="Helvetica-Bold")
    body_s   = S("B",  fontSize=8,  textColor=DARK,  spaceAfter=3, leading=12,     fontName="Helvetica")
    prompt_s = S("P",  fontSize=8,  textColor=colors.HexColor("#333333"), fontName="Helvetica-Oblique",
                       spaceAfter=4, leading=12, leftIndent=12, rightIndent=12)
    model_s  = S("M",  fontSize=9,  fontName="Helvetica-Bold", spaceAfter=2)
    resp_s   = S("R",  fontSize=7.5, textColor=DARK, leading=11, spaceAfter=4,
                       fontName="Helvetica", leftIndent=16, rightIndent=8)
    score_s  = S("Sc", fontSize=7.5, textColor=GREY, fontName="Helvetica-Oblique", spaceAfter=6)
    note_s   = S("N",  fontSize=7,  textColor=GREY,  fontName="Helvetica-Oblique", spaceAfter=8)
    winner_s = S("W",  fontSize=9,  textColor=DARK,  fontName="Helvetica-Bold",    spaceAfter=6)

    def tbl_style(extra=None):
        cmds = [
            ("BACKGROUND",     (0,0),  (-1,0),  BLUE),
            ("TEXTCOLOR",      (0,0),  (-1,0),  WHITE),
            ("FONTNAME",       (0,0),  (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),  (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1),  (-1,-1), [WHITE, LIGHT]),
            ("GRID",           (0,0),  (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
            ("VALIGN",         (0,0),  (-1,-1), "MIDDLE"),
            ("LEFTPADDING",    (0,0),  (-1,-1), 6),
            ("RIGHTPADDING",   (0,0),  (-1,-1), 6),
            ("TOPPADDING",     (0,0),  (-1,-1), 4),
            ("BOTTOMPADDING",  (0,0),  (-1,-1), 4),
        ] + (extra or [])
        return TableStyle(cmds)

    def section(title):
        return [Spacer(1, 0.3*cm), Paragraph(f"  {title}", sec_s), Spacer(1, 0.2*cm)]

    # Aggregate stats
    model_agg = {m[2]: {"scores": [], "times": [], "errors": 0} for m in models}
    for r in results:
        lbl = r["model_label"]
        if lbl not in model_agg:
            continue
        if "scores" in r:
            model_agg[lbl]["scores"].append(r["scores"]["final"])
        model_agg[lbl]["times"].append(r["elapsed"])
        if r["response"].startswith("ERROR"):
            model_agg[lbl]["errors"] += 1

    ranked = sorted([
        (e[2],
         round(sum(model_agg[e[2]]["scores"]) / len(model_agg[e[2]]["scores"]), 3)
         if model_agg[e[2]]["scores"] else 0.0,
         round(sum(model_agg[e[2]]["times"])  / len(model_agg[e[2]]["times"]),  1)
         if model_agg[e[2]]["times"]  else 0.0,
         model_agg[e[2]]["errors"],
         compute_monthly_cost(e[3], e[4], e[0]),
        ) for e in models
    ], key=lambda x: -x[1])

    budget_ranked = sorted(
        [(lbl, avg, mo, round(avg / (1 + math.log10(1 + mo)), 3) if mo > 0 else avg)
         for lbl, avg, _, __, mo in ranked],
        key=lambda x: -x[3],
    )

    story = []

    # Cover
    story += [
        Spacer(1, 1.5*cm),
        Paragraph("WaterSec AI Agent", title_s),
        Paragraph("Model Benchmark &amp; Cost Analysis Report", sub_s),
        Paragraph(
            f"Generated: {run_date}  |  "
            f"Data: {'Real IoT sensor data (' + str(len(_DF)) + ' rows)' if _DATA_LOADED else 'Pre-computed fallback'}",
            date_s,
        ),
        HRFlowable(width="100%", thickness=1.5, color=BLUE),
        Spacer(1, 0.4*cm),
    ]

    # Sec 1 — Key status
    story += section("1  Groq Key Status")
    kd = [["Key", "Status"]]
    for i in range(len(_raw_keys)):
        nm = f"KEY_{i+1}"
        if i >= len(_groq_keys):
            st, c = "Not configured", RED
        elif _groq_cooldowns.get(i, 0) > time.time():
            st, c = "In cooldown", RED
        else:
            st, c = "Available", GREEN
        kd.append([nm, Paragraph(f'<font color="#{c.hexval()[2:]}"><b>{st}</b></font>', body_s)])
    kt = Table(kd, colWidths=[4*cm, 12*cm])
    kt.setStyle(tbl_style())
    story += [kt, Spacer(1, 0.15*cm),
              Paragraph("Keys rotate automatically — rate-limited keys get a 60s cooldown before retry.", note_s)]

    # Sec 2 — Overview
    story += section("2  Overview")
    n_prompts = len(set(r["prompt_id"] for r in results))
    story.append(Paragraph(
        f"Compares <b>{len(models)}</b> LLM candidates on <b>{n_prompts}</b> water domain prompts. "
        f"<b>Real sensor data is injected into each prompt</b> so models answer with actual numbers. "
        f"Scoring: <b>S = 0.25\u00b7R + 0.30\u00b7C + 0.25\u00b7I + 0.20\u00b7D \u2212 0.5\u00b7H</b> "
        f"(Relevance, Correctness, Insight, Domain 1\u20135; Hallucination 0\u20132).",
        body_s,
    ))

    # Sec 3 — Models
    story += section("3  Models Tested")
    md = [["Model", "Type", "Speed", "In $/1M", "Out $/1M", "Monthly $"]]
    for provider, _, lbl, inp, out, spd, _ in models:
        c  = BLUE if "Groq" in lbl else GREEN
        mo = compute_monthly_cost(inp, out, provider)
        md.append([
            Paragraph(f'<font color="#{c.hexval()[2:]}"><b>{lbl}</b></font>', body_s),
            "API" if provider == "groq" else "Local",
            f"{spd} t/s" if spd else "CPU",
            f"${inp:.3f}", f"${out:.3f}", f"${mo:.2f}",
        ])
    mt = Table(md, colWidths=[5.5*cm, 2*cm, 2.5*cm, 2*cm, 2.2*cm, 2.3*cm])
    mt.setStyle(tbl_style())
    story += [mt, Spacer(1, 0.15*cm),
              Paragraph("Monthly: 1,500 queries, avg 900 in + 700 out tokens. Hardware amortised 36mo/$3,000.", note_s)]

    # Sec 4 — Prompt results
    story += section("4  Prompt-by-Prompt Results  [real sensor data injected]")
    cur_cat = None
    all_prompts = _build_prompts()
    for pid in sorted(set(r["prompt_id"] for r in results)):
        pr = [x for x in results if x["prompt_id"] == pid]
        if not pr:
            continue
        pmeta = next((p for p in all_prompts if p["id"] == pid), {})
        cat   = pmeta.get("category", "")
        diff  = pmeta.get("difficulty", "")
        full  = pmeta.get("prompt", "")
        base  = full.split("\n\nDATA FROM WATERSEC")[0]

        block = []
        if cat != cur_cat:
            block.append(Paragraph(f"{cat} Prompts", sub2_s))
            cur_cat = cat

        block.append(Paragraph(f"<b>[{pid}] {diff}</b>",
                                S("pid", fontSize=8.5, fontName="Helvetica-Bold", textColor=DARK, spaceAfter=2)))
        block.append(Paragraph(base[:300] + ("..." if len(base) > 300 else ""), prompt_s))
        if "\n\nDATA FROM WATERSEC" in full:
            block.append(Paragraph("\u2713 Real sensor data injected into this prompt", note_s))

        for r in pr:
            lbl  = r["model_label"]
            c    = BLUE if "Groq" in lbl else GREEN
            raw  = r["response"]
            if raw == "[NO_RESPONSE]":
                raw = "[NO_RESPONSE — Groq returned empty content (silent per-model rate-limit). Auto-scored 0.]"
            disp = raw[:1500] + ("... [truncated]" if len(raw) > 1500 else "")
            block.append(Paragraph(
                f'<font color="#{c.hexval()[2:]}"><b>{lbl}</b></font>'
                f'<font color="#999999">  \u00b7  {r["elapsed"]:.1f}s</font>', model_s,
            ))
            safe = _sanitize(disp).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            block.append(Paragraph(safe, resp_s))
            if "scores" in r:
                s = r["scores"]
                block.append(Paragraph(
                    f'R:{s["relevance"]:.0f}  C:{s["correctness"]:.0f}  '
                    f'I:{s["insight"]:.0f}  D:{s["domain"]:.0f}  H:{s["hallucination"]:.0f}'
                    f'  \u2192  <b>Final: {s["final"]:.3f} / 5.0</b>', score_s,
                ))
        block += [HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#DDDDDD")),
                  Spacer(1, 0.1*cm)]
        story.append(KeepTogether(block[:5]))
        story += block[5:]

    # Sec 5 — Scorecard
    story += [PageBreak()] + section("5  Summary Scorecard")
    sd = [["Model", "Avg Score", "Avg Time", "Errors", "Monthly $"]]
    for lbl, avg, avg_t, errs, mo in ranked:
        c  = BLUE if "Groq" in lbl else GREEN
        ec = RED  if errs > 0    else GREEN
        sd.append([
            Paragraph(f'<font color="#{c.hexval()[2:]}"><b>{lbl}</b></font>', body_s),
            f"{avg:.3f}", f"{avg_t:.1f}s",
            Paragraph(f'<font color="#{ec.hexval()[2:]}"><b>{errs}</b></font>', body_s),
            f"${mo:.2f}",
        ])
    st2 = Table(sd, colWidths=[6*cm, 2.5*cm, 2.5*cm, 2*cm, 3.5*cm])
    st2.setStyle(tbl_style())
    story += [st2, Spacer(1, 0.3*cm)]

    # Sec 6 — Cost-quality
    story += section("6  Cost\u2013Quality Recommendation")
    story.append(Paragraph(
        "Budget score = quality \u00f7 (1 + log\u2081\u2080(1 + monthly cost)).  Higher = better quality per dollar.",
        body_s,
    ))
    story.append(Spacer(1, 0.2*cm))
    cd = [["", "Model", "Avg Quality", "Monthly $", "Budget Score"]]
    for i, (lbl, avg, mo, budget) in enumerate(budget_ranked):
        c     = BLUE if "Groq" in lbl else GREEN
        medal = "\u2605" if i == 0 else ("2" if i == 1 else "3" if i == 2 else "")
        cd.append([
            Paragraph(f'<font color="#{GOLD.hexval()[2:]}"><b>{medal}</b></font>', body_s) if medal else "",
            Paragraph(f'<font color="#{c.hexval()[2:]}"><b>{lbl}</b></font>', body_s),
            f"{avg:.3f}", f"${mo:.2f}",
            Paragraph(f"<b>{budget:.3f}</b>", body_s),
        ])
    ct = Table(cd, colWidths=[0.8*cm, 5.5*cm, 2.8*cm, 2.8*cm, 3.1*cm])
    ct.setStyle(tbl_style(extra=[("BACKGROUND", (0,1), (-1,1), colors.HexColor("#FFF8E1"))]))
    story += [ct, Spacer(1, 0.25*cm)]
    winner = budget_ranked[0][0] if budget_ranked else "N/A"
    story.append(Paragraph(
        f"\u2605  <b>Recommendation:</b> <b>{winner}</b> offers the best quality-to-cost balance. "
        f"For on-premises deployments, prefer the highest-scoring local model.",
        winner_s,
    ))

    # Sec 7 — Fallback chain
    story += section("7  Proposed Production Fallback Chain")
    items = []
    for i, (lbl, *_) in enumerate(budget_ranked):
        reason = ("Best budget score" if i == 0
                  else "Local fallback — zero cost / data privacy" if "Local" in lbl
                  else "Rate-limit fallback")
        c = BLUE if "Groq" in lbl else GREEN
        items.append(ListItem(
            Paragraph(f'<font color="#{c.hexval()[2:]}"><b>{lbl}</b></font> — {reason}', body_s),
            bulletColor=c,
        ))
    story.append(ListFlowable(items, bulletType="1", leftIndent=20))
    story += [Spacer(1, 0.15*cm),
              Paragraph("Groq keys rotate automatically across KEY_1\u2013KEY_4. "
                        "Reorder _PROVIDERS in llm_client.py to change the agent default.", note_s)]

    def _hf(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(2*cm, H - 1.2*cm, "WaterSec Model Benchmark Report — v5.0")
        canvas.drawRightString(W - 2*cm, H - 1.2*cm, run_date)
        canvas.drawString(2*cm, 1.2*cm, f"Page {doc.page}")
        canvas.drawRightString(W - 2*cm, 1.2*cm,
                               "Data: real WaterSec IoT sensors" if _DATA_LOADED else "Data: pre-computed fallback")
        canvas.setStrokeColor(BLUE)
        canvas.setLineWidth(0.5)
        canvas.line(2*cm, H - 1.4*cm, W - 2*cm, H - 1.4*cm)
        canvas.line(2*cm, 1.5*cm,     W - 2*cm, 1.5*cm)
        canvas.restoreState()

    SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title="WaterSec Model Benchmark Report",
        author="WaterSec AI Agent",
    ).build(story, onFirstPage=_hf, onLaterPages=_hf)
    print(f"  [\u2713] PDF report \u2192 {pdf_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def auto_score_response(prompt: str, response: str) -> dict:
    """
    Score a response automatically using GPT-OSS 120B as judge.
    Falls back to Llama 3.3 70B if 120B returns empty.
    Single attempt per model — no retry loop.
    """
    judge_prompt = (
        f"You are a STRICT benchmark judge for a water IoT analytics system. Be harsh and precise.\n\n"
        f"SCORING RULES:\n"
        f"relevance  1-5: 1=ignores question, 3=partial, 5=fully answers every part\n"
        f"correctness 1-5: CHECK NUMBERS carefully. 1=wrong numbers/logic, 3=mostly right, 5=all correct. "
        f"If the model refuses to answer or gives irrelevant output, score 1.\n"
        f"insight    1-5: 1=just restates data, 3=some analysis, 5=actionable expert insight\n"
        f"domain     1-5: 1=ignores water/IoT context, 3=uses some domain terms, 5=expert water knowledge. "
        f"Do NOT give 5 just because it mentions water. Require correct domain reasoning.\n"
        f"hallucination 0-2: 0=all claims supported by data, 1=minor unsupported claim, "
        f"2=fabricated numbers or wrong facts stated confidently. Be suspicious of invented statistics.\n\n"
        f"QUESTION: {prompt[:400]}\n\n"
        f"RESPONSE: {response[:700]}\n\n"
        f"Think: does the response actually answer the question correctly with right numbers? "
        f"Is the domain score truly deserved or just surface-level water terminology?\n"
        f"Reply format — ONLY these 5 integers space-separated, nothing else: R C I D H"
    )

    def _try_judge(model_id: str) -> str:
        """Single attempt at judge model. Returns raw string or raises."""
        try:
            raw, elapsed = _call_groq(
                model_id,
                [{"role": "user", "content": judge_prompt}],
                max_tokens=100,  # was 20 — gives room to avoid finish_reason=length
                temperature=0.0,
            )
            if raw == "[NO_RESPONSE]":
                raise ValueError(f"Judge {model_id} returned NO_RESPONSE")
            return raw
        except RuntimeError:
            raise
        except Exception as e:
            raise ValueError(f"Judge {model_id} failed: {e}") from e

    import re

    # Try primary judge (120B)
    raw = None
    # Skip 120B — consistently returns empty (finish_reason=length)
    for model_id in ["llama-3.3-70b-versatile"]:  # removed "openai/gpt-oss-120b"
        try:
            raw = _try_judge(model_id)
            break
        except (RuntimeError, ValueError) as e:
            print(f"  [judge] {model_id} unavailable: {e}")
            continue

    if raw is None:
        print(f"  [auto-score] All judge models failed — using fallback 1/1/1/1/2")
        return {
            "relevance": 1, "correctness": 1, "insight": 1,
            "domain": 1, "hallucination": 2,
            "final": compute_score({"relevance":1,"correctness":1,"insight":1,"domain":1,"hallucination":2}),
        }

    try:
        raw_clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        parts = raw_clean.strip().split()
        # Expect exactly 5 integers
        r, c, i, d, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        r, c, i, d = [max(1, min(5, x)) for x in [r, c, i, d]]
        h = max(0, min(2, h))
        scores = {"relevance": r, "correctness": c, "insight": i, "domain": d, "hallucination": h}
        scores["final"] = compute_score(scores)
        print(f"  [auto] R:{r} C:{c} I:{i} D:{d} H:{h} -> {scores['final']:.3f}")
        return scores
    except Exception as e:
        print(f"  [auto-score parse failed: {e}] raw='{raw[:100]}' -> fallback 1/1/1/1/2")
        return {
            "relevance": 1, "correctness": 1, "insight": 1,
            "domain": 1, "hallucination": 2,
            "final": compute_score({"relevance":1,"correctness":1,"insight":1,"domain":1,"hallucination":2}),
        }


def run_benchmark(selected_models, selected_prompts, do_score=True, do_auto=False):
    results  = []
    run_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    print("\n" + "\u2550"*80)
    print("  WATERSEC MODEL BENCHMARK v5.0")
    print(f"  Models: {len(selected_models)}   Prompts: {len(selected_prompts)}   "
          f"Groq keys: {len(_groq_keys)}   Data loaded: {_DATA_LOADED}   Scoring: {'ON' if do_score else 'OFF'}")
    print("\u2550"*80)

    for p in selected_prompts:
        has_data = "\n\nDATA FROM WATERSEC" in p["prompt"]
        print(f"\n{'█'*80}")
        print(f"  [{p['id']}] {p['category'].upper()} | {p['difficulty'].upper()}"
              + ("  [+data]" if has_data else "  [no data]"))
        base = p["prompt"].split("\n\nDATA FROM WATERSEC")[0]
        print(f"  {base[:120]}{'...' if len(base)>120 else ''}")
        print("\u2500"*80)

        for model_entry in selected_models:
            _, _, label, *__ = model_entry
            print(f"\n  \u25b6 {label}")
            print("\u00b7"*60)
            response, elapsed = run_prompt(model_entry, p["prompt"])
            print(response[:800] + ("..." if len(response) > 800 else ""))
            print(f"\n  \u23f1  {elapsed:.1f}s")
            print("\u00b7"*60)

            entry = {"prompt_id": p["id"], "model_label": label,
                     "response": response, "elapsed": elapsed}
            if response in ("[NO_RESPONSE]",) or response.startswith("ERROR"):
                entry["scores"] = {
                    "relevance": 1, "correctness": 1, "insight": 1,
                    "domain": 1, "hallucination": 2,
                    "final": compute_score({"relevance":1,"correctness":1,
                                           "insight":1,"domain":1,"hallucination":2}),
                }
                tag = "NO_RESPONSE" if response == "[NO_RESPONSE]" else "ERROR"
                print(f"  [auto-scored {tag} -> {entry['scores']['final']:.3f}]")
            elif do_auto:
                entry["scores"] = auto_score_response(p["prompt"], response)
            elif do_score:
                entry["scores"] = score_response(label)
            results.append(entry)

        if do_score:
            input("\n  [Press Enter to continue...]\n")

    return results, run_date


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _line1 = "  WaterSec Model Benchmark v5.0"
    _line2 = f"  Groq keys: {len(_groq_keys)} configured   Data loaded: {_DATA_LOADED}"
    _width = max(len(_line1), len(_line2)) + 2
    _bar   = "\u2550" * _width
    print(f"\n  \u2554{_bar}\u2557")
    print(f"  \u2551{_line1:<{_width}}\u2551")
    print(f"  \u2551{_line2:<{_width}}\u2551")
    print(f"  \u255a{_bar}\u255d\n")

    if not _groq_keys:
        print("  \u26a0  No Groq keys — only local Ollama models will run.\n")

    # Auto-pull local models
    for _, model_id, label, *_ in [m for m in REGISTRY if m[0] == "ollama"]:
        ensure_model_pulled(model_id)

    print("\n  Available models:")
    for i, (provider, _, lbl, inp, out, spd, _) in enumerate(REGISTRY):
        mo = compute_monthly_cost(inp, out, provider)
        print(f"    {i+1}. {lbl:<38} {str(spd)+' t/s' if spd else 'CPU':<12} ${mo:.2f}/mo")

    print("\n  Model selection:")
    print("    A) All models")
    print("    B) Cloud only (Groq)")
    print("    C) Local only (Ollama)")
    print("    D) Custom (e.g. 1,3,4)")
    mc = input("\n  Choice: ").strip().upper()

    if   mc == "A": selected_models = list(REGISTRY)
    elif mc == "B": selected_models = [m for m in REGISTRY if m[0] == "groq"]
    elif mc == "C": selected_models = [m for m in REGISTRY if m[0] == "ollama"]
    elif mc == "D":
        idxs = input("  Numbers: ").strip()
        selected_models = [REGISTRY[int(i)-1] for i in idxs.split(",")]
    else:
        selected_models = list(REGISTRY)

    ALL_PROMPTS = _build_prompts()

    print("\n  Prompt selection:")
    print("    A) All (14 prompts)")
    print("    B) Standard only (S1–S5)")
    print("    C) Advanced only (A1–A6)")
    print("    D) Tool-Use only (T1–T3)")
    print("    E) Quick test (S1 + A5 + T1)")
    print("    F) Custom numbers")
    pc = input("\n  Choice: ").strip().upper()

    if   pc == "A": selected_prompts = ALL_PROMPTS
    elif pc == "B": selected_prompts = [p for p in ALL_PROMPTS if p["category"] == "Standard"]
    elif pc == "C": selected_prompts = [p for p in ALL_PROMPTS if p["category"] == "Advanced"]
    elif pc == "D": selected_prompts = [p for p in ALL_PROMPTS if p["category"] == "Tool-Use"]
    elif pc == "E": selected_prompts = [ALL_PROMPTS[0], ALL_PROMPTS[9], ALL_PROMPTS[11]]
    elif pc == "F":
        print("  Prompts:")
        for i, p in enumerate(ALL_PROMPTS):
            print(f"    {i+1}. [{p['id']}] {p['difficulty']:<8} "
                  f"{p['prompt'].split(chr(10))[0][:65]}...")
        idxs = input("  Numbers: ").strip()
        selected_prompts = [ALL_PROMPTS[int(i)-1] for i in idxs.split(",")]
    else:
        selected_prompts = ALL_PROMPTS

    print("\n  Scoring mode:")
    print("    H) Human (you score each response manually)")
    print("    A) Auto  (GPT-OSS 120B scores each response — no human input needed)")
    print("    N) None  (run only, no scoring — scorecard will show 0.000)")
    score_mode = input("\n  Choice: ").strip().upper()
    do_score   = score_mode == "H"
    do_auto    = score_mode == "A"

    results, run_date = run_benchmark(selected_models, selected_prompts, do_score, do_auto)

    ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    json_path  = os.path.join(output_dir, f"benchmark_results_{ts}.json")
    pdf_path   = os.path.join(output_dir, f"benchmark_report_{ts}.pdf")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"run_date": run_date, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n  [\u2713] Raw results  \u2192 {json_path}")

    print("  [\u00b7] Generating PDF...")
    generate_pdf(results, selected_models, run_date, pdf_path)
    print(f"\n  Done! Open: {pdf_path}\n")