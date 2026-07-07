import os
import json
import time

from google import genai
from google.genai import types

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential


EVAL_PROMPT_TEMPLATE = """
You are an expert technical interviewer. Evaluate the candidate's transcript against the expected answer.
Focus on semantic meaning, not exact keyword matches.

Expected Answer: {expected_answer}

Candidate Transcript: {transcript}


Provide the evaluation strictly in the following JSON schema:

{{
    "semantic_score": <int 0-100 based on core concept coverage>,
    "matched_concepts": [
        <list of key concepts the candidate successfully explained>
    ],
    "missing_concepts": [
        <list of key concepts the candidate missed>
    ],
    "feedback": "<A concise, 1-2 sentence constructive feedback string>"
}}
"""


# ---------------- Gemini Text Evaluation ----------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=2,
        min=2,
        max=10
    ),
    reraise=True
)
def _call_gemini_with_retry(prompt, gemini_key):
    print("[Gemini] Attempting text generation...")

    client = genai.Client(
        api_key=gemini_key
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    return json.loads(response.text)


# ---------------- Groq Backup ----------------

def _call_groq_fallback(prompt, groq_key):
    print("[Fallback] Routing text request to Groq...")

    client = Groq(
        api_key=groq_key
    )

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    return json.loads(
        completion.choices[0].message.content
    )


# ---------------- Semantic Evaluation ----------------

def evaluate_semantic_robust(
        transcript,
        expected_answer,
        gemini_key,
        groq_key
):
    """
    Evaluates answer quality.
    Gemini primary + Groq fallback.
    """

    prompt = EVAL_PROMPT_TEMPLATE.format(
        expected_answer=expected_answer,
        transcript=transcript
    )

    try:
        return _call_gemini_with_retry(
            prompt,
            gemini_key
        )

    except Exception as gemini_err:
        print(
            f"[Warning] Gemini text unavailable: {gemini_err}"
        )

        try:
            return _call_groq_fallback(
                prompt,
                groq_key
            )

        except Exception as groq_err:
            print(
                f"[Error] Both text providers failed: {groq_err}"
            )

            return {
                "semantic_score": 0,
                "matched_concepts": [],
                "missing_concepts": [
                    "System Overload"
                ],
                "feedback":
                    "Evaluation failed temporarily due to upstream API issues."
            }


# ---------------- Video Behaviour Evaluation ----------------

def evaluate_visual_behavior(
        video_path,
        gemini_key
):
    """
    Analyze interview video:
    - eye contact
    - confidence
    - posture
    - facial expression
    """

    print(
        f"[Gemini] Uploading {video_path} for behavioral analysis..."
    )

    client = genai.Client(
        api_key=gemini_key
    )

    video_file = client.files.upload(
        file=video_path
    )


    # wait until Gemini finishes video processing

    while video_file.state.name == "PROCESSING":

        print(
            "Waiting for video processing..."
        )

        time.sleep(2)

        video_file = client.files.get(
            name=video_file.name
        )


    if video_file.state.name != "ACTIVE":

        raise Exception(
            f"Video processing failed: {video_file.state.name}"
        )


    response = client.models.generate_content(
        model="gemini-2.5-flash",

        contents=[
            video_file,

            """
            Analyze this interview video.

            Evaluate:
            - eye contact
            - confidence
            - posture
            - facial expressions

            Return only JSON:

            {
              "eye_contact_score": 0-10,
              "confidence_score": 0-10,
              "posture_feedback": "",
              "facial_expression_feedback": "",
              "overall_feedback": ""
            }
            """
        ],

        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )


    return json.loads(
        response.text
    )