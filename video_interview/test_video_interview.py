import json
import os
from interview_evaluation import evaluate_video_interview
from report_formatter import generate_report

# Ensure your keys are set in your terminal before running, or paste them here directly for local testing
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "your-gemini-key-here")
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "your-groq-key-here")

question = "Explain the difference between supervised and unsupervised learning."

expected_answer = """
Supervised learning uses labelled data to train a model.
It is used for classification and regression tasks.
Unsupervised learning uses unlabelled data and finds hidden patterns.
It is commonly used for clustering and dimensionality reduction.
"""

# IMPORTANT: Ensure this path points to a real test video you recorded
video_path = "sample_videos/test.mp4" 

try:
    result = evaluate_video_interview(
        question=question,
        expected_answer=expected_answer,
        video_path=video_path,
        model_name="tiny"
    )
    print("\n=== FINAL EVALUATION RESULT ===")
    print(generate_report(result))
except Exception as e:
    print(f"Test failed: {e}")