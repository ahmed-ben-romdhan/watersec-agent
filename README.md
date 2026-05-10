# WaterSec AI Agent

## Setup
```bash
pip install -r requirements.txt
```

## Configure API keys
Copy `.env.example` to `.env` and fill in your keys:
- Groq: https://console.groq.com → API Keys
- Download Ollama and respective models : 

## Add your data
Place these 4 CSV files in the `data/` folder:
- gym_consumption_data.csv
- customerA_consumption.csv
- customerB_consumption.csv
- customerC_consumption.csv

## Run
```bash
python app.py
```
Then open the Gradio URL shown in the terminal.
