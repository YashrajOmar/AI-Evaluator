def stars(score):
    filled = round(score / 2)
    return "⭐" * filled + "☆" * (5 - filled)


def generate_report(result):
    behavior = result.get("behavioral_analysis", {})

    report = f"""
====================================
🎯 Interview Evaluation Report
====================================

Question:
{result.get('question')}

------------------------------------
📊 Overall Performance
------------------------------------

⭐ Final Score: {result.get('overall_score')} / 10

🧠 Knowledge Score:
{result.get('semantic_score')} / 100

🗣 Speaking Fluency:
{result.get('fluency_score')} / 10


------------------------------------
🧠 Answer Analysis
------------------------------------

✅ What you did well:
"""

    for item in result.get("matched_concepts", []):
        report += f"• {item}\n"

    report += """

❌ Areas to improve:
"""

    missing = result.get("missing_concepts", [])

    if missing:
        for item in missing:
            report += f"• {item}\n"
    else:
        report += "• Great! No important concepts missed.\n"


    report += f"""

💡 Feedback:
{result.get('semantic_feedback')}


------------------------------------
🗣 Speaking Performance
------------------------------------

Words Spoken:
{result.get('word_count')}

Speaking Speed:
{result.get('speaking_rate_wpm')} words/min


------------------------------------
🎥 Body Language
------------------------------------

👀 Eye Contact:
{stars(behavior.get('eye_contact_score',0))}
({behavior.get('eye_contact_score',0)}/10)


💪 Confidence:
{stars(behavior.get('confidence_score',0))}
({behavior.get('confidence_score',0)}/10)


🧍 Posture:
{behavior.get('posture_feedback','N/A')}


🙂 Facial Expression:
{behavior.get('facial_expression_feedback','N/A')}


------------------------------------
🏁 Final Summary
------------------------------------

{behavior.get('overall_feedback','')}
"""

    return report
