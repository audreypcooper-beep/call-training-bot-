import json
import os
from openai import OpenAI
from pathlib import Path

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
RESULTS_DIR = Path("data/results")

def load_scenario(file_id: str, scenario_id: str = "scenario_1") -> dict:
    result_path = RESULTS_DIR / f"{file_id}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"No result found for file_id: {file_id}")

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", {}).get("scenarios", [])
    for scenario in scenarios:
        if scenario.get("id") == scenario_id:
            return scenario
    if scenarios:
        return scenarios[0]
    raise ValueError("No scenarios found")

def start_roleplay(file_id: str, scenario_id: str = "scenario_1") -> dict:
    scenario = load_scenario(file_id, scenario_id)

    system_prompt = f"""
You are roleplaying as a customer in a call center training simulation.

### Your Persona
Name: {scenario['caller_persona']['name']}
Emotional state: {scenario['caller_persona']['emotional_state']}
Background: {scenario['caller_persona']['background']}

### Situation
{scenario['situation']}

### Your Goal
{scenario['caller_goal']}

### Rules
- Stay fully in character as the customer.
- Respond naturally and realistically.
- Do not break character or mention that you are an AI.
- Keep responses concise (1-4 sentences) unless the situation requires more detail.
- Match the emotional state described above.
"""

    opening_line = scenario.get("opening_line", "Hello, I need some help.")

    return {
        "file_id": file_id,
        "scenario_id": scenario.get("id"),
        "scenario_title": scenario.get("title"),
        "difficulty": scenario.get("difficulty"),
        "system_prompt": system_prompt,
        "messages": [{"role": "assistant", "content": opening_line}],
        "opening_message": opening_line
    }

def continue_roleplay(system_prompt: str, messages: list, user_message: str, model: str = "gpt-4o") -> dict:
    updated_messages = messages + [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, *updated_messages],
        temperature=0.8,
        max_tokens=300
    )

    ai_reply = response.choices[0].message.content
    updated_messages.append({"role": "assistant", "content": ai_reply})

    return {"reply": ai_reply, "messages": updated_messages}

def evaluate_roleplay(system_prompt: str, messages: list, scenario: dict, model: str = "gpt-4o") -> dict:
    conversation_log = []
    for msg in messages:
        role = "Agent (Trainee)" if msg["role"] == "user" else "Caller (AI)"
        conversation_log.append(f"{role}: {msg['content']}")
    conversation_text = "\n".join(conversation_log)

    evaluation_prompt = f"""
You are an expert call center trainer and quality coach.

Evaluate the trainee (the Agent) based on the roleplay conversation below.

### Scenario Information
Title: {scenario.get('title')}
Difficulty: {scenario.get('difficulty')}
Caller Goal: {scenario.get('caller_goal')}
Success Criteria:
{chr(10).join('- ' + c for c in scenario.get('success_criteria', []))}

### Full Conversation
{conversation_text}

### Evaluation Instructions
Provide a structured assessment with the following JSON format:

{{
  "overall_score": 0-100,
  "scores": {{
    "greeting_and_tone": 0-10,
    "active_listening": 0-10,
    "problem_understanding": 0-10,
    "solution_quality": 0-10,
    "empathy_and_soft_skills": 0-10,
    "clarity_and_communication": 0-10,
    "closing": 0-10
  }},
  "strengths": ["Specific strength 1", "Specific strength 2"],
  "areas_for_improvement": ["Specific improvement 1", "Specific improvement 2"],
  "detailed_feedback": "2-4 paragraphs of constructive, professional coaching feedback",
  "would_resolve_in_real_call": true,
  "key_coaching_tip": "One clear, actionable tip for the trainee"
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise and constructive call center trainer. Always respond with valid JSON only."},
            {"role": "user", "content": evaluation_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)
