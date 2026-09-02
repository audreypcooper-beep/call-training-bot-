import json
import os
from openai import OpenAI
from pathlib import Path

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def load_prompt(filename: str) -> str:
    with open(PROMPTS_DIR / filename, "r", encoding="utf-8") as f:
        return f.read()

def analyze_transcript(transcript: str, model: str = "gpt-4o") -> dict:
    prompt = load_prompt("analysis_prompt.txt").format(transcript=transcript)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise call center analyst. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)

def generate_scenarios(analysis: dict, model: str = "gpt-4o") -> dict:
    prompt = load_prompt("scenario_prompt.txt").format(analysis_json=json.dumps(analysis, indent=2))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert training scenario designer. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    return json.loads(response.choices[0].message.content)

def process_call(transcript: str) -> dict:
    analysis = analyze_transcript(transcript)
    scenarios = generate_scenarios(analysis)
    return {"analysis": analysis, "scenarios": scenarios}
