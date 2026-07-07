import os
import sys
import json
import base64
import asyncio
import traceback
import subprocess
from typing import List
import httpx
from pydantic import BaseModel

from dotenv import load_dotenv
from groq import AsyncGroq
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

# Load environment variables from the .env file in this directory
load_dotenv()

# Initialize Groq Client
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

# --- SYSTEM CONFIGURATION ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# --- APP INITIALIZATION ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPERS ---
def compile_and_run_cpp_sync(cpp_file: str, exe_file: str) -> str:
    """Compiles and runs C++ code synchronously."""
    try:
        comp_proc = subprocess.run(
            ["g++", cpp_file, "-o", exe_file],
            capture_output=True, text=True
        )
        if comp_proc.returncode != 0:
            return "Compilation Error:\n" + comp_proc.stderr
        
        try:
            run_proc = subprocess.run(
                [exe_file],
                capture_output=True, text=True, timeout=5.0
            )
            return run_proc.stdout + run_proc.stderr
        except subprocess.TimeoutExpired:
            return "Execution Error: Process timed out after 5 seconds."
            
    except Exception as e:
        traceback.print_exc()
        return f"Execution Error: {repr(e)}"

def analyze_face_sync(frame):
    """Runs lightweight DeepFace analysis."""
    result = DeepFace.analyze(
        frame, 
        actions=['emotion'], 
        enforce_detection=False,
        detector_backend='opencv', 
        align=False 
    )
    if isinstance(result, list):
        result = result[0]
    return result.get('dominant_emotion', 'neutral')

# --- 1. FACIAL FEEDBACK WEBSOCKET ---
@app.websocket("/ws/inference")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    is_processing = False  
    
    try:
        while True:
            b64_data = await websocket.receive_text()
            
            if is_processing:
                continue

            is_processing = True

            async def process_frame(data):
                confidence_score = 0.5
                try:
                    if "," in data:
                        data = data.split(",")[1]
                    
                    data += "=" * ((4 - len(data) % 4) % 4)
                        
                    img_bytes = base64.b64decode(data)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        dominant_emotion = await asyncio.to_thread(analyze_face_sync, frame)
                        
                        emotion_map = {
                            "happy": 0.95, "neutral": 0.75, "surprise": 0.60,
                            "sad": 0.40, "angry": 0.30, "fear": 0.20, "disgust": 0.10
                        }
                        confidence_score = emotion_map.get(dominant_emotion, 0.5)
                        
                        # ---> THE MISSING LINE HAS BEEN ADDED HERE <---
                        print(f"Detected: {dominant_emotion} | Score: {confidence_score}")
                    else:
                        print("WARNING: Decoded frame is None.")
                        
                except Exception as e:
                    print(f"FAILED TO ANALYZE FRAME: {e}")
                
                finally:
                    try:
                        await websocket.send_json({"confidence": float(confidence_score)})
                    except RuntimeError:
                        pass
                    except Exception as e:
                        print(f"Websocket send error: {e}")

            task = asyncio.create_task(process_frame(b64_data))
            
            def on_done(t):
                nonlocal is_processing
                is_processing = False
                
            task.add_done_callback(on_done)

    except WebSocketDisconnect:
        print("Client disconnected normally.")
    except Exception as e:
        print(f"Unexpected websocket error: {e}")

# --- 2. C++ CODE EXECUTION ENDPOINT ---
class CodePayload(BaseModel):
    code: str

@app.post("/execute")
async def execute_code(payload: CodePayload):
    if not payload.code.strip():
        return {"output": "Error: Code cannot be empty."}

    wandbox_url = "https://wandbox.org/api/compile.json"
    sandbox_payload = {
        "compiler": "gcc-head", 
        "code": payload.code,
        "save": False
    }

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(wandbox_url, json=sandbox_payload, timeout=10.0)
            
        if response.status_code == 200:
            result = response.json()
            compiler_error = result.get("compiler_error", "")
            if compiler_error.strip():
                return {"output": f"Compilation Error:\n{compiler_error}"}
            
            output = result.get("program_message", "Executed successfully with no output.")
            return {"output": output}
        else:
            return {"output": f"Sandbox Engine Error: {response.status_code} - {response.text}"}
            
    except Exception as e:
        print(f"Sandbox connection failed: {e}")
        return {"output": "Execution service is temporarily unavailable."}

# --- 3. EVALUATION ENDPOINT ---
class TranscriptLine(BaseModel):
    role: str
    text: str

class EvaluationPayload(BaseModel):
    role: str
    duration_seconds: int
    average_composure: float
    transcript: List[TranscriptLine]
    system_prompt_instruction: str 

@app.post("/evaluate-session")
async def evaluate_session(payload: EvaluationPayload):
    composure_percentage = round(payload.average_composure * 100)
    transcript_text = "\n".join([f"{msg.role}: {msg.text}" for msg in payload.transcript])
    
    if not transcript_text.strip():
        return {
            "overall_score": 0, "content_score": 0, "composure_score": composure_percentage,
            "technical_feedback": "No conversation recorded.",
            "behavioral_feedback": "No conversation recorded.",
            "key_strengths": ["None"], "areas_for_improvement": ["Candidate did not speak."]
        }

    try:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": payload.system_prompt_instruction},
                {"role": "user", "content": f"Interview Transcript:\n{transcript_text}"}
            ]
        )
        
        evaluation = json.loads(response.choices[0].message.content)
        
        return {
            "overall_score": evaluation.get("overall_score", 0),
            "content_score": evaluation.get("content_score", 0),
            "composure_score": evaluation.get("composure_score", composure_percentage),
            "technical_feedback": evaluation.get("technical_feedback", "N/A"),
            "behavioral_feedback": evaluation.get("behavioral_feedback", "N/A"),
            "key_strengths": evaluation.get("key_strengths", []),
            "areas_for_improvement": evaluation.get("areas_for_improvement", [])
        }

    except Exception as e:
        print("LLM Evaluation Failed:", e)
        traceback.print_exc()
        return {
            "overall_score": 0, "content_score": 0, "composure_score": composure_percentage,
            "technical_feedback": "API Error: Could not generate feedback.",
            "behavioral_feedback": f"Error Details: {str(e)}",
            "key_strengths": ["Error processing transcript"], "areas_for_improvement": ["Check backend logs"]
        }