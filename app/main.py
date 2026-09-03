import os
import shutil
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.transcription import transcribe_audio
from app.analysis import process_call
from app.roleplay import start_roleplay, continue_roleplay, evaluate_roleplay, load_scenario

app = FastAPI(title="Call Training Bot")

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent

UPLOAD_DIR = BASE_DIR / "data" / "audio"
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"
RESULTS_DIR = BASE_DIR / "data" / "results"
STATIC_DIR = BASE_DIR / "static"

for folder in [UPLOAD_DIR, TRANSCRIPT_DIR, RESULTS_DIR, STATIC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class RoleplayStartRequest(BaseModel):
    file_id: str
    scenario_id: Optional[str] = "scenario_1"


class RoleplayContinueRequest(BaseModel):
    system_prompt: str
    messages: List[dict]
    user_message: str


class RoleplayEvaluateRequest(BaseModel):
    file_id: str
    scenario_id: str = "scenario_1"
    system_prompt: str
    messages: List[dict]


@app.get("/", response_class=HTMLResponse)
def home():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return FileResponse(index_file)


@app.post("/process-call")
async def process_call_endpoint(file: UploadFile = File(...)):
    allowed = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(allowed)}")

    file_id = str(uuid.uuid4())
    audio_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        transcript = transcribe_audio(str(audio_path))
        (TRANSCRIPT_DIR / f"{file_id}.txt").write_text(transcript, encoding="utf-8")

        result = process_call(transcript)

        full_result = {
            "file_id": file_id,
            "original_filename": file.filename,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "audio_path": str(audio_path),
            "transcript": transcript,
            "analysis": result["analysis"],
            "scenarios": result["scenarios"],
        }

        (RESULTS_DIR / f"{file_id}.json").write_text(
            json.dumps(full_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return JSONResponse(content=full_result)

    except Exception as e:
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)
        raise HTTPException(500, str(e))


@app.get("/calls")
async def list_processed_calls():
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            analysis = data.get("analysis", {})
            reason = analysis.get("reason_for_call", {})
            outcome = analysis.get("outcome", {})
            results.append({
                "file_id": data.get("file_id"),
                "original_filename": data.get("original_filename"),
                "processed_at": data.get("processed_at"),
                "reason_for_call": reason.get("primary_category"),
                "detailed_reason": reason.get("detailed_reason"),
                "outcome": outcome.get("status"),
                "agent_name": analysis.get("agent", {}).get("name"),
                "has_scenarios": bool(data.get("scenarios", {}).get("scenarios")),
            })
        except Exception:
            continue
    return {"total_calls": len(results), "calls": results}


@app.get("/calls/{file_id}")
async def get_call_result(file_id: str):
    path = RESULTS_DIR / f"{file_id}.json"
    if not path.exists():
        raise HTTPException(404, f"No call found with file_id: {file_id}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/roleplay/start")
async def roleplay_start(request: RoleplayStartRequest):
    try:
        return start_roleplay(request.file_id, request.scenario_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/roleplay/continue")
async def roleplay_continue(request: RoleplayContinueRequest):
    try:
        return continue_roleplay(
            system_prompt=request.system_prompt,
            messages=request.messages,
            user_message=request.user_message,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/roleplay/evaluate")
async def roleplay_evaluate(request: RoleplayEvaluateRequest):
    try:
        scenario = load_scenario(request.file_id, request.scenario_id)
        evaluation = evaluate_roleplay(
            system_prompt=request.system_prompt,
            messages=request.messages,
            scenario=scenario,
        )
        return {
            "file_id": request.file_id,
            "scenario_id": request.scenario_id,
            "evaluation": evaluation,
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
