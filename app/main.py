import shutil
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from app.transcription import transcribe_audio
from app.analysis import process_call
from app.roleplay import start_roleplay, continue_roleplay, evaluate_roleplay, load_scenario

load_dotenv()

app = FastAPI(title="Call Training Bot")

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent

UPLOAD_DIR = BASE_DIR / "data" / "audio"
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"
RESULTS_DIR = BASE_DIR / "data" / "results"
STATIC_DIR = BASE_DIR / "static"

for folder in [UPLOAD_DIR, TRANSCRIPT_DIR, RESULTS_DIR, STATIC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Call Training Bot</title>
  <style>
    body { font-family: system-ui, sans-serif; background:#f8fafc; color:#1e293b; padding:20px; }
    .container { max-width:900px; margin:0 auto; }
    .card { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; margin-bottom:20px; }
    button { background:#2563eb; color:#fff; border:0; padding:10px 18px; border-radius:8px; cursor:pointer; }
    button:disabled { background:#94a3b8; }
    .danger { background:#dc2626; }
    input[type="file"], #user-input { width:100%; margin:10px 0; padding:10px; }
    .status { color:#64748b; margin-top:10px; }
    #chat-box { height:320px; overflow:auto; border:1px solid #e2e8f0; border-radius:8px; padding:12px; background:#f1f5f9; }
    .message { margin:10px 0; padding:10px; border-radius:10px; max-width:80%; }
    .agent { background:#dbeafe; margin-left:auto; }
    .caller { background:#fff; border:1px solid #e2e8f0; }
    .hidden { display:none; }
    .call-item { display:flex; justify-content:space-between; gap:12px; padding:12px 0; border-bottom:1px solid #e2e8f0; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Call Training Bot</h1>
    <p>Upload a call, analyze it, then practice the roleplay.</p>

    <div class="card">
      <h2>1. Upload Call Recording</h2>
      <input type="file" id="audio-file" accept="audio/*" />
      <button onclick="uploadCall()" id="upload-btn">Process Call</button>
      <div class="status" id="upload-status"></div>
    </div>

    <div class="card">
      <h2>2. Processed Calls</h2>
      <button onclick="loadCalls()">Refresh List</button>
      <div id="calls-list"></div>
    </div>

    <div class="card hidden" id="roleplay-section">
      <h2>3. Roleplay Practice</h2>
      <div id="scenario-info"></div>
      <div id="chat-box"></div>
      <input type="text" id="user-input" placeholder="Type your response as the Agent..." />
      <button onclick="sendMessage()" id="send-btn">Send</button>
      <button onclick="endRoleplay()" class="danger">End & Score</button>
      <div class="status" id="roleplay-status"></div>
    </div>

    <div class="card hidden" id="evaluation-section">
      <h2>4. Performance Feedback</h2>
      <div id="evaluation-content"></div>
    </div>
  </div>

  <script>
    const API = window.location.origin;
    let currentSession = null;

    function getErrorMessage(err) {
      if (!err) return "Unknown error";
      if (typeof err === "string") return err;
      if (err.detail) return typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
      return err.message || JSON.stringify(err);
    }

    async function readJson(res) {
      const text = await res.text();
      try { return JSON.parse(text); }
      catch (e) { throw new Error(text || "Invalid server response"); }
    }

    async function uploadCall() {
      const fileInput = document.getElementById("audio-file");
      const status = document.getElementById("upload-status");
      const btn = document.getElementById("upload-btn");
      if (!fileInput.files.length) {
        status.textContent = "Please select an audio file first.";
        return;
      }
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      btn.disabled = true;
      status.textContent = "Uploading and processing... this may take 1-3 minutes.";
      try {
        const res = await fetch(API + "/process-call", { method: "POST", body: formData });
        const data = await readJson(res);
        if (!res.ok) throw new Error(getErrorMessage(data));
        status.textContent = "Success! File ID: " + data.file_id;
        loadCalls();
      } catch (err) {
        status.textContent = "Error: " + getErrorMessage(err);
      } finally {
        btn.disabled = false;
      }
    }

    async function loadCalls() {
      const list = document.getElementById("calls-list");
      list.innerHTML = "Loading...";
      try {
        const res = await fetch(API + "/calls");
        const data = await readJson(res);
        if (!res.ok) throw new Error(getErrorMessage(data));
        if (!data.calls || data.calls.length === 0) {
          list.innerHTML = "<p>No calls processed yet.</p>";
          return;
        }
        list.innerHTML = data.calls.map(call => `
          <div class="call-item">
            <div>
              <strong>${call.original_filename || "Unknown"}</strong><br/>
              ${call.reason_for_call || "—"} · ${call.outcome || "—"}
            </div>
            <button onclick="startRoleplay('${call.file_id}')">Practice</button>
          </div>
        `).join("");
      } catch (err) {
        list.innerHTML = "Failed to load calls: " + getErrorMessage(err);
      }
    }

    async function startRoleplay(fileId) {
      const status = document.getElementById("roleplay-status");
      status.textContent = "Starting roleplay...";
      try {
        const res = await fetch(API + "/roleplay/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_id: fileId, scenario_id: "scenario_1" })
        });
        const data = await readJson(res);
        if (!res.ok) throw new Error(getErrorMessage(data));
        currentSession = data;
        document.getElementById("roleplay-section").classList.remove("hidden");
        document.getElementById("evaluation-section").classList.add("hidden");
        document.getElementById("scenario-info").innerHTML = "<strong>" + (data.scenario_title || "Scenario") + "</strong>";
        document.getElementById("chat-box").innerHTML = "";
        appendMessage("caller", data.opening_message || "Hello, I need help.");
        status.textContent = "Roleplay started. You are the Agent.";
      } catch (err) {
        status.textContent = "Error: " + getErrorMessage(err);
      }
    }

    async function sendMessage() {
      const input = document.getElementById("user-input");
      const text = input.value.trim();
      if (!text || !currentSession) return;
      appendMessage("agent", text);
      input.value = "";
      try {
        const res = await fetch(API + "/roleplay/continue", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            system_prompt: currentSession.system_prompt,
            messages: currentSession.messages,
            user_message: text
          })
        });
        const data = await readJson(res);
        if (!res.ok) throw new Error(getErrorMessage(data));
        currentSession.messages = data.messages;
        appendMessage("caller", data.reply);
      } catch (err) {
        appendMessage("caller", "[Error] " + getErrorMessage(err));
      }
    }

    function appendMessage(role, text) {
      const box = document.getElementById("chat-box");
      const div = document.createElement("div");
      div.className = "message " + role;
      div.textContent = (role === "agent" ? "You: " : "Caller: ") + (text || "");
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    async function endRoleplay() {
      if (!currentSession) return;
      const status = document.getElementById("roleplay-status");
      status.textContent = "Evaluating...";
      try {
        const res = await fetch(API + "/roleplay/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file_id: currentSession.file_id,
            scenario_id: currentSession.scenario_id || "scenario_1",
            system_prompt: currentSession.system_prompt,
            messages: currentSession.messages
          })
        });
        const data = await readJson(res);
        if (!res.ok) throw new Error(getErrorMessage(data));
        const ev = data.evaluation || {};
        document.getElementById("evaluation-section").classList.remove("hidden");
        document.getElementById("evaluation-content").innerHTML =
          "<p><strong>Score:</strong> " + (ev.overall_score || 0) + "/100</p>" +
          "<p>" + (ev.detailed_feedback || "") + "</p>" +
          "<p><strong>Tip:</strong> " + (ev.key_coaching_tip || "") + "</p>";
        status.textContent = "Evaluation complete.";
      } catch (err) {
        status.textContent = "Evaluation failed: " + getErrorMessage(err);
      }
    }

    document.getElementById("user-input").addEventListener("keydown", function(e) {
      if (e.key === "Enter") sendMessage();
    });

    loadCalls();
  </script>
</body>
</html>
"""


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
def root():
    for path in [
        STATIC_DIR / "index.html",
        HERE / "static" / "index.html",
        Path("static/index.html"),
        Path("index.html"),
    ]:
        if path.exists():
            return FileResponse(path)
    return HTMLResponse(INDEX_HTML)


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
