# WaterSec AI Agent

Conversational water monitoring agent powered by Llama 3.3 70B via Groq API.

## Setup

pip install -r requirements.txt

## Configure API keys

Copy .env.example to .env and fill in your keys:
- Groq: https://console.groq.com → API Keys
- Download Ollama and pull the required models:
  ollama pull qwen2.5:3b
  ollama pull mistral:7b-instruct

## Add your data

Place these 4 CSV files in the data/ folder:
- gym_consumption_data.csv
- customerA_consumption.csv
- customerB_consumption.csv
- customerC_consumption.csv

## Run

python app.py

Then open the Gradio URL shown in the terminal.
