import re
import os
import subprocess
import json
from typing import Dict, List, Optional
from video_behavior import evaluate_semantic_robust, evaluate_visual_behavior

FILLER_WORDS = {"um", "uh", "like", "basically", "actually", "literally", "you know", "i mean", "so", "well", "hmm"}

def clean_text(text: str) -> str:
    if not text: return ""
    return re.sub(r"\s+", " ", text.lower().strip())

def calculate_fluency_score(transcript: str, duration_seconds: Optional[float] = None) -> Dict:
    text = clean_text(transcript)
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    filler_count = sum(text.count(filler) for filler in FILLER_WORDS)

    speaking_rate = None
    if duration_seconds and duration_seconds > 0:
        duration_minutes = duration_seconds / 60
        speaking_rate = round(word_count / duration_minutes, 2)

    score = 10
    if word_count < 20: score -= 3
    elif word_count < 40: score -= 1

    if filler_count > 10: score -= 3
    elif filler_count > 5: score -= 2
    elif filler_count > 2: score -= 1

    if speaking_rate:
        if speaking_rate < 80 or speaking_rate > 180: score -= 2

    return {
        "fluency_score": max(0, min(10, score)),
        "word_count": word_count,
        "filler_word_count": filler_count,
        "speaking_rate_wpm": speaking_rate
    }

def evaluate_interview(question: str, expected_answer: str, transcript: str, duration_seconds: Optional[float] = None) -> Dict:
    # Pull keys from environment
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")

    # Run the new LLM Semantic Evaluation
    content_result = evaluate_semantic_robust(transcript, expected_answer, gemini_key, groq_key)
    fluency_result = calculate_fluency_score(transcript, duration_seconds)

    # Calculate overall based on the LLM's 0-100 semantic score
    semantic_score_normalized = content_result.get("semantic_score", 0) / 10
    overall_score = round((0.7 * semantic_score_normalized) + (0.3 * fluency_result["fluency_score"]), 2)

    return {
        "question": question,
        "transcript": transcript,
        "semantic_score": content_result.get("semantic_score", 0),
        "fluency_score": fluency_result["fluency_score"],
        "overall_score": overall_score,
        "matched_concepts": content_result.get("matched_concepts", []),
        "missing_concepts": content_result.get("missing_concepts", []),
        "semantic_feedback": content_result.get("feedback", ""),
        "word_count": fluency_result["word_count"],
        "filler_word_count": fluency_result["filler_word_count"],
        "speaking_rate_wpm": fluency_result["speaking_rate_wpm"],
    }

def get_media_duration_seconds(file_path: str) -> float:
    if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(result.stdout.strip())

def extract_audio_from_video(video_path: str, output_audio_path: str = None) -> str:
    if not os.path.exists(video_path): raise FileNotFoundError(f"Video file not found: {video_path}")
    if output_audio_path is None:
        base_name = os.path.splitext(video_path)[0]
        output_audio_path = base_name + "_audio.wav"
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return output_audio_path

def transcribe_audio(audio_path: str, model_name: str = "tiny") -> str:
    import whisper
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"].strip()

def transcribe_video(video_path: str, model_name: str = "tiny") -> str:
    audio_path = extract_audio_from_video(video_path)
    return transcribe_audio(audio_path, model_name=model_name)

def evaluate_video_interview(question: str, expected_answer: str, video_path: str, model_name: str = "tiny") -> Dict:
    duration_seconds = get_media_duration_seconds(video_path)
    transcript = transcribe_video(video_path, model_name=model_name)

    # 1. Evaluate Text and Audio fluency
    result = evaluate_interview(question, expected_answer, transcript, duration_seconds)

    # 2. Evaluate Visual Behavior
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        behavior_result = evaluate_visual_behavior(video_path, gemini_key)
        result["behavioral_analysis"] = behavior_result
    except Exception as e:
        print(f"[Warning] Visual behavior analysis failed: {e}")
        result["behavioral_analysis"] = {"error": str(e)}

    result["video_path"] = video_path
    result["duration_seconds"] = round(duration_seconds, 2)
    result["transcription_model"] = model_name

    return result