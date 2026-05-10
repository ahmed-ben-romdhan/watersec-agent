"""
llm_client.py — Groq-1 → Groq-2 → Groq-3 → Groq-4 → Groq-5-Fast → Local fallback chain.
Token-aware: counts tokens before sending, truncates history if needed.
"""
import os
import time
import tiktoken
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
_groq1 = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY_1", "dummy"))
_groq2 = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY_2", "dummy"))
_groq3 = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY_3", "dummy"))
_groq4 = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY_4", "dummy"))
_groq5 = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY_5", "dummy"))
_ollama = OpenAI(base_url="http://localhost:11434/v1",      api_key="ollama")

_PROVIDERS = [
    (_groq1,  "llama-3.3-70b-versatile", "Groq-1",       8000),
    (_groq2,  "llama-3.3-70b-versatile", "Groq-2",       8000),
    (_groq3,  "llama-3.3-70b-versatile", "Groq-3",       8000),
    (_groq4,  "llama-3.3-70b-versatile", "Groq-4",       8000),
    (_groq5,  "llama-3.1-8b-instant",    "Groq-5-Fast",  8000),
    (_ollama, "qwen2.5:3b",              "Local-Qwen-3B", 32000),
]

# Per-key cooldowns
_cooldowns: dict[int, float] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
_current_provider_idx = 0

# Token counter
_enc = tiktoken.get_encoding("cl100k_base")

def _count_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        total += len(_enc.encode(content))
    return total

def _truncate(messages: list, max_tokens: int) -> list:
    """Keep system prompt + trim oldest messages until under max_tokens."""
    system = [messages[0]]
    rest   = messages[1:]
    while _count_tokens(system + rest) > max_tokens and len(rest) > 2:
        rest = rest[2:]  # drop oldest user+assistant pair
    return system + rest

# ── Main caller ───────────────────────────────────────────────────────────────
def call_llm(messages: list, max_tokens: int = 1500, temperature: float = 0.2) -> tuple[str, str]:
    global _current_provider_idx

    now = time.time()
    ordered = sorted(range(len(_PROVIDERS)), key=lambda i: _cooldowns.get(i, 0))

    for idx in ordered:
        if _cooldowns.get(idx, 0) > now:
            print(f"[llm] {_PROVIDERS[idx][2]} in cooldown — skipping")
            continue

        client, model, name, token_limit = _PROVIDERS[idx]

        safe_messages = _truncate(messages, token_limit - max_tokens - 200)
        n_tokens = _count_tokens(safe_messages)
        print(f"[llm] Trying {name} ({model}) — {n_tokens} tokens...")

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=safe_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            _current_provider_idx = idx
            print(f"[llm] Response from {name}")
            return resp.choices[0].message.content, name

        except RateLimitError as e:
            cooldown = 90 if "429" in str(e) else 60
            print(f"[llm] {name} rate limit — {cooldown}s cooldown")
            _cooldowns[idx] = now + cooldown
            continue

        except APIError as e:
            if "413" in str(e) or "too large" in str(e).lower():
                print(f"[llm] {name} 413 too large — skipping for 10s")
                _cooldowns[idx] = now + 10
            else:
                print(f"[llm] {name} API error: {e}")
            continue

        except Exception as e:
            print(f"[llm] {name} unexpected: {e}")
            continue

    raise RuntimeError("All providers failed — check Groq keys and Ollama status.")

def get_active_provider() -> str:
    return _PROVIDERS[_current_provider_idx][2]