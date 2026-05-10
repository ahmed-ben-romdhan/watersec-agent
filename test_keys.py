from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

groq1 = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY_1"))
groq2 = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY_2"))

for name, client in [("Groq-1", groq1), ("Groq-2", groq2)]:
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        print(f"{name} OK: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"{name} FAILED: {type(e).__name__}: {e}")