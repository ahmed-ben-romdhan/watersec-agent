"""
app.py — WaterSec AI Agent entry point.
Run with: python app.py
"""

import os
import sys
import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv

# ── Auto-start Ollama if not already running ──────────────────────────────────
import subprocess
import time
import requests

def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        print("[ollama] Already running.")
        return
    except Exception:
        pass

    print("[ollama] Starting Ollama in background...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    for _ in range(15):
        time.sleep(1)
        try:
            requests.get("http://localhost:11434/api/tags", timeout=2)
            print("[ollama] Ready.")
            return
        except Exception:
            pass
    print("[ollama] WARNING: Ollama did not start in time — local fallback may fail.")

ensure_ollama_running()


# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

if not os.environ.get("GROQ_API_KEY_1") and not os.environ.get("GROQ_API_KEY_2"):
    print("ERROR: No API keys found. Add GROQ_API_KEY_1 and GROQ_API_KEY_2 to your .env file.")
    sys.exit(1)

# ── Load and clean data ───────────────────────────────────────────────────────
print("[app] Loading datasets...")
import tools
from data_loader import load_and_clean, get_summary

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    print(f"ERROR: '{DATA_DIR}/' folder not found. Create it and add your 4 CSV files.")
    sys.exit(1)

df = load_and_clean(DATA_DIR)
tools.DF = df
DATASET_SUMMARY = get_summary(df)

from agent import run_agent, SYSTEM_PROMPT
import agent as _agent
_agent.SYSTEM_PROMPT += f"\n\nLIVE DATASET SUMMARY:\n{DATASET_SUMMARY}"

print("[app] Agent ready.\n")

# ── Global chart storage (persists across calls) ──────────────────────────────
_chart_files: list[str] = []


def respond(user_message: str, history: list):
    """
    Returns: (history, main_chart_visible, gallery_update, chart_files_state)
    """
    global _chart_files
    
    if not user_message.strip():
        return history, gr.update(visible=False), gr.update(), _chart_files

    # Convert Gradio history to tuples for agent
    history_tuples = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            history_tuples.append((history[i]["content"], history[i+1]["content"]))
            i += 2
        else:
            i += 1

    text, chart, provider = run_agent(user_message, history_tuples)
    from llm_client import _PROVIDERS
    model_name = next((m for _, m, n, _ in _PROVIDERS if n == provider), provider)
    label = f"[{provider} · {model_name}]"
    full_response = f"{text}\n\n*{label}*"

    # ── Save chart to gallery if generated ────────────────────────────────────
    main_chart_visible = gr.update(visible=False)
    
    if chart is not None:
        try:
            import plotly.io as pio
            import tempfile
            from datetime import datetime
            timestamp = datetime.now().strftime("%H%M%S")
            title_slug = (chart.layout.title.text or "Chart")[:30].replace(" ", "_").replace("/", "_")
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", 
                delete=False,
                prefix=f"{timestamp}_{title_slug}_"
            )
            pio.write_image(chart, tmp.name, width=900, height=500)
            _chart_files.append(tmp.name)
            main_chart_visible = gr.update(value=tmp.name, visible=True)
        except Exception as e:
            print(f"[app] Chart PNG export skipped: {e}")

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": full_response})
    
    return history, main_chart_visible, _chart_files, _chart_files


EXAMPLE_PROMPTS = [
    "What is the total water consumption at the gym last month?",
    "Compare hot vs cold water usage across all gym shower cabins",
    "Plot the daily consumption trend for customerC over the past 3 months",
    "Are there any anomalies in the residential data this year?",
    "Which customer has the highest average daily consumption?",
    "Detect usage patterns in the residential data — do flushes lead to sink usage?",
    "Generate a heatmap of gym consumption by hour of day and day of week",
    "Compare customerA and customerB total consumption side by side",
]


def generate_dataset_overview():
    rows = []
    profiles = {
        "gym":       ("Shower block",        "8 (4 cabins × hot/cold)"),
        "customera": ("Office toilet bloc",   "4 aggregated"),
        "customerb": ("Sanitary bloc",        "1 aggregated"),
        "customerc": ("Residential home",     "7 (flush/sink/tap)"),
    }
    for customer, grp in df.groupby("customer"):
        profile, sensors = profiles.get(customer.lower(), ("Unknown", "?"))
        start     = grp["data_time"].min().strftime("%b %Y")
        end       = grp["data_time"].max().strftime("%b %Y")
        total_L   = grp["consumption_L"].sum()
        daily_avg = total_L / max((grp["data_time"].max() - grp["data_time"].min()).days, 1)
        n_events  = len(grp)
        avg_flow  = grp["flow_rate_Lpm"].mean()

        rows.append(
            f"| **{customer}** | {profile} | {sensors} | "
            f"{start} – {end} | "
            f"{total_L:,.0f} L | "
            f"{daily_avg:,.1f} L/day | "
            f"{n_events:,} | "
            f"{avg_flow:.2f} L/min |"
        )

    header = (
        "### Dataset Overview\n"
        "| Customer | Profile | Sensors | Period | Total | Avg/Day | Events | Avg Flow |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows)


def select_chart_from_gallery(evt: gr.SelectData, chart_files: list):
    """When user clicks a gallery image, display it in the main chart area."""
    if chart_files and evt.index < len(chart_files):
        return gr.update(value=chart_files[evt.index], visible=True)
    return gr.update(visible=False)


with gr.Blocks(title="WaterSec AI Agent") as demo:
    gr.Markdown("## WaterSec AI Agent")
    gr.Markdown("**Conversational water consumption analysis**")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Agent",
                height=520,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask anything about water consumption...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Markdown("**Example prompts:**")
            for prompt in EXAMPLE_PROMPTS:
                gr.Button(prompt, size="sm").click(
                    fn=lambda p=prompt: p,
                    outputs=msg_box,
                )

        with gr.Column(scale=2):
            chart_out = gr.Image(
                label="Chart",
                visible=False,
                type="filepath",
            )
            chart_gallery = gr.Gallery(
                label="Generated Charts (click to view)",
                columns=2,
                height=250,
                visible=True,
                object_fit="contain",
            )
            gr.Markdown(value=generate_dataset_overview())

    # ── State ────────────────────────────────────────────────────────────────
    chart_store = gr.State([])

    # ── Event handlers ───────────────────────────────────────────────────────
    def submit(message, history, charts):
        return respond(message, history)

    msg_box.submit(
        submit, 
        [msg_box, chatbot, chart_store], 
        [chatbot, chart_out, chart_gallery, chart_store]
    ).then(fn=lambda: "", outputs=msg_box)
    
    send_btn.click(
        submit, 
        [msg_box, chatbot, chart_store], 
        [chatbot, chart_out, chart_gallery, chart_store]
    ).then(fn=lambda: "", outputs=msg_box)

    # Click on gallery → show that chart in main view
    chart_gallery.select(
        select_chart_from_gallery,
        chart_store,
        chart_out,
    )

    gr.Markdown(
        "_Data: gym (Apr 2024–May 2026) · Customer A (May–Nov 2025) · "
        "Customer B (Oct 2025–May 2026) · Customer C (Mar 2024–Nov 2025)_"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        inbrowser=True,
        allowed_paths=["."],
        css="""
            .gradio-container {
                background-image: url('/gradio_api/file=wallpaper.jpg');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
        """,
    )