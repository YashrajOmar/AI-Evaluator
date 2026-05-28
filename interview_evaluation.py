import re
import json
from typing import Dict, List, Optional


FILLER_WORDS = {
    "um", "uh", "like", "basically", "actually", "literally",
    "you know", "i mean", "so", "well", "hmm"
}

STOPWORDS = {
    "the", "is", "am", "are", "was", "were", "a", "an", "and", "or", "of",
    "to", "in", "for", "on", "with", "as", "by", "this", "that", "it",
    "be", "can", "will", "from", "at", "which", "uses", "use"
}


def clean_text(text: str) -> str:
    """Convert text to lowercase and remove extra spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_keywords(text: str) -> List[str]:
    """
    Extract useful keywords from expected answer or transcript.
    This is a simple baseline method.
    """
    text = clean_text(text)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)

    keywords = []
    for word in words:
        if word not in STOPWORDS and word not in keywords:
            keywords.append(word)

    return keywords


def calculate_content_score(expected_answer: str, transcript: str) -> Dict:
    """
    Compare candidate transcript with expected answer using keyword coverage.
    Score is out of 10.
    """
    expected_keywords = extract_keywords(expected_answer)
    transcript_text = clean_text(transcript)

    if not expected_keywords:
        return {
            "content_score": 0,
            "matched_keywords": [],
            "missing_keywords": [],
            "keyword_coverage": 0
        }

    matched_keywords = [
        keyword for keyword in expected_keywords
        if keyword in transcript_text
    ]

    missing_keywords = [
        keyword for keyword in expected_keywords
        if keyword not in transcript_text
    ]

    coverage = len(matched_keywords) / len(expected_keywords)
    content_score = round(coverage * 10, 2)

    return {
        "content_score": content_score,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "keyword_coverage": round(coverage, 2)
    }


def calculate_fluency_score(
    transcript: str,
    duration_seconds: Optional[float] = None
) -> Dict:
    """
    Calculate basic communication metrics:
    - word count
    - speaking rate
    - filler words
    - fluency score
    """
    text = clean_text(transcript)
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    filler_count = 0
    for filler in FILLER_WORDS:
        filler_count += text.count(filler)

    speaking_rate = None
    if duration_seconds and duration_seconds > 0:
        duration_minutes = duration_seconds / 60
        speaking_rate = round(word_count / duration_minutes, 2)

    score = 10

    if word_count < 20:
        score -= 3
    elif word_count < 40:
        score -= 1

    if filler_count > 10:
        score -= 3
    elif filler_count > 5:
        score -= 2
    elif filler_count > 2:
        score -= 1

    if speaking_rate:
        if speaking_rate < 80:
            score -= 2
        elif speaking_rate > 180:
            score -= 2

    fluency_score = max(0, min(10, score))

    return {
        "fluency_score": fluency_score,
        "word_count": word_count,
        "filler_word_count": filler_count,
        "speaking_rate_wpm": speaking_rate
    }


def generate_feedback(
    content_result: Dict,
    fluency_result: Dict
) -> str:
    """Generate simple feedback based on scores."""
    feedback = []

    content_score = content_result["content_score"]
    fluency_score = fluency_result["fluency_score"]

    if content_score >= 8:
        feedback.append("The answer is highly relevant and covers most expected points.")
    elif content_score >= 5:
        feedback.append("The answer is partially correct but misses some important points.")
    else:
        feedback.append("The answer needs improvement and misses many expected concepts.")

    missing = content_result["missing_keywords"]
    if missing:
        feedback.append(
            "Missing concepts include: " + ", ".join(missing[:5]) + "."
        )

    if fluency_score >= 8:
        feedback.append("Communication is clear and fluent.")
    elif fluency_score >= 5:
        feedback.append("Communication is understandable but can be more fluent.")
    else:
        feedback.append("The response has fluency issues and should be more structured.")

    if fluency_result["filler_word_count"] > 2:
        feedback.append("Try to reduce filler words such as um, uh, like, and basically.")

    return " ".join(feedback)


def evaluate_interview(
    question: str,
    expected_answer: str,
    transcript: str,
    duration_seconds: Optional[float] = None
) -> Dict:
    """
    Main function for interview evaluation.
    For now, transcript is given manually.
    Later, transcript will come from speech-to-text.
    """
    content_result = calculate_content_score(expected_answer, transcript)
    fluency_result = calculate_fluency_score(transcript, duration_seconds)

    overall_score = round(
        (0.7 * content_result["content_score"]) +
        (0.3 * fluency_result["fluency_score"]),
        2
    )

    return {
        "question": question,
        "transcript": transcript,
        "content_score": content_result["content_score"],
        "fluency_score": fluency_result["fluency_score"],
        "overall_score": overall_score,
        "matched_keywords": content_result["matched_keywords"],
        "missing_keywords": content_result["missing_keywords"],
        "word_count": fluency_result["word_count"],
        "filler_word_count": fluency_result["filler_word_count"],
        "speaking_rate_wpm": fluency_result["speaking_rate_wpm"],
        "feedback": generate_feedback(content_result, fluency_result)
    }


if __name__ == "__main__":
    sample_question = "Explain the difference between supervised and unsupervised learning."

    sample_expected_answer = """
    Supervised learning uses labelled data to train a model.
    It is used for classification and regression tasks.
    Unsupervised learning uses unlabelled data and finds hidden patterns.
    It is commonly used for clustering and dimensionality reduction.
    """

    sample_transcript = """
    Supervised learning uses labelled data where the model learns from examples.
    Unsupervised learning uses unlabelled data and tries to find patterns.
    For example, clustering is an unsupervised learning task.
    """

    result = evaluate_interview(
        question=sample_question,
        expected_answer=sample_expected_answer,
        transcript=sample_transcript,
        duration_seconds=35
    )

    print(json.dumps(result, indent=4))