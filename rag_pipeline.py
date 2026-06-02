"""
rag_pipeline.py
Retrieval-Augmented Generation pipeline for question paper generation.

Flow:
  topic / difficulty  →  retrieve relevant context from FAISS
                       →  feed context + prompt to LLM
                       →  structured question paper output
"""

from __future__ import annotations

from vector_store import load_vector_store, retrieve_chunks
from langchain_community.vectorstores import FAISS
from balanced_retrieval import balanced_retrieve


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty profiles
# ─────────────────────────────────────────────────────────────────────────────

DIFFICULTY_PROFILES = {
    "easy": {
        "description": "Recall and basic understanding",
        "mcq_instruction": "Test direct recall of definitions, facts, and basic concepts.",
        "short_instruction": "Ask students to define or describe concepts in their own words.",
        "long_instruction": "Ask students to explain a concept with a simple example.",
        "bloom_levels": "Remember, Understand",
    },
    "medium": {
        "description": "Application and comprehension",
        "mcq_instruction": "Test understanding by asking students to apply concepts to simple scenarios.",
        "short_instruction": "Ask students to solve small problems or compare two related concepts.",
        "long_instruction": "Ask students to apply an algorithm to a given input and explain each step.",
        "bloom_levels": "Apply, Analyse",
    },
    "hard": {
        "description": "Analysis, design, and evaluation",
        "mcq_instruction": "Test deep understanding with tricky edge cases, complexity analysis, or design trade-offs.",
        "short_instruction": "Ask students to analyse time/space complexity or identify flaws in an approach.",
        "long_instruction": "Ask students to design an algorithm from scratch, prove correctness, or compare multiple approaches with justification.",
        "bloom_levels": "Evaluate, Create",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Default IIT-style rubric criteria
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RUBRIC = [
    "Conceptual clarity",
    "Application ability",
    "Time/space complexity",
]

ALL_RUBRIC_CRITERIA = {
    "Conceptual clarity":    "Questions must test precise understanding of definitions and theoretical foundations.",
    "Application ability":   "Questions must require applying concepts to solve new or unseen problems.",
    "Time/space complexity": "At least one question per section must require Big-O analysis.",
    "Algorithm tracing":     "Include at least one question requiring step-by-step dry run of an algorithm.",
    "Algorithm design":      "Include questions that require writing pseudocode or designing an algorithm from scratch.",
    "Comparative analysis":  "Include questions that ask students to compare approaches and justify their choice.",
    "Edge case handling":    "Questions should probe boundary conditions and failure modes.",
    "Correctness proof":     "Include at least one question requiring informal proof or loop invariant analysis.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(
    topic: str,
    context: str,
    num_mcq: int = 5,
    num_short: int = 3,
    num_long: int = 2,
    difficulty: str = "medium",
    rubric_criteria: list[str] | None = None,
) -> str:

    profile = DIFFICULTY_PROFILES.get(difficulty, DIFFICULTY_PROFILES["medium"])

    # Build rubric block
    if rubric_criteria is None:
        rubric_criteria = DEFAULT_RUBRIC
    rubric_lines = "\n".join(
        f"  • {c}: {ALL_RUBRIC_CRITERIA.get(c, '')}"
        for c in rubric_criteria
        if c in ALL_RUBRIC_CRITERIA
    )
    rubric_block = f"""
QUESTION STYLE PREFERENCES (use these to shape question types):
{rubric_lines}
""".strip()

    return f"""
You are an expert university professor at a premier institution (IIT-level) creating a rigorous examination paper.

The following CONTEXT has been retrieved from the course material
(lecture notes, slides, or textbooks) provided by the professor.
Base your questions STRICTLY on this context — do not add outside knowledge.

──────────────────────────────
CONTEXT:
{context}
──────────────────────────────

DIFFICULTY LEVEL: {difficulty.upper()} — {profile['description']}
Bloom's Taxonomy target: {profile['bloom_levels']}

{rubric_block}

SECTION-SPECIFIC INSTRUCTIONS:
- MCQ:          {profile['mcq_instruction']}
- Short answer: {profile['short_instruction']}
- Long answer:  {profile['long_instruction']}

MARKING SCHEME REQUIREMENT (mandatory for every short and long answer question):
For EVERY short answer question, immediately after the question write:
Marking scheme:
  - <specific criterion from this question> : <marks>
  - <specific criterion from this question> : <marks>
  - <specific criterion from this question> : <marks>
  (must add up to 5 marks exactly)

For EVERY long answer question, immediately after the question write:
Marking scheme:
  - <specific criterion from this question> : <marks>
  - <specific criterion from this question> : <marks>
  - <specific criterion from this question> : <marks>
  - <specific criterion from this question> : <marks>
  (must add up to 10 marks exactly)

Criteria must be SPECIFIC to each question — not generic labels like "clarity" or "understanding".
Example for a question about AVL rotations:
  - Balance factor definition correct : 2
  - LL and RR rotation steps shown : 3
  - LR and RL rotation explained : 3
  - Time complexity stated : 2

TASK:
Generate a question paper on the topic: "{topic}"

Format EXACTLY as shown below — no extra commentary outside the template:

===== QUESTION PAPER =====
Topic: {topic}
Difficulty: {difficulty.capitalize()}
Total Marks: {num_mcq * 1 + num_short * 5 + num_long * 10}

--- Section A: Multiple Choice Questions ({num_mcq} Questions × 1 Mark = {num_mcq} Marks) ---
Q1. <question>
   (a) <option>  (b) <option>  (c) <option>  (d) <option>
   Answer: <correct option letter>

[repeat for Q2 … Q{num_mcq}]

--- Section B: Short Answer Questions ({num_short} Questions × 5 Marks = {num_short * 5} Marks) ---
Q{num_mcq + 1}. <question>  [5 marks]
Marking scheme:
  - <specific criterion> : <marks>
  - <specific criterion> : <marks>
  - <specific criterion> : <marks>
  (total: 5 marks)

[repeat for remaining short answer questions]

--- Section C: Long Answer / Essay Questions ({num_long} Questions × 10 Marks = {num_long * 10} Marks) ---
Q{num_mcq + num_short + 1}. <question>  [10 marks]
Marking scheme:
  - <criterion 1> : <marks>
  - <criterion 2> : <marks>
  - <criterion 3> : <marks>

[repeat for remaining long answer questions]
===========================
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# LLM caller
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Call Groq API (free).
    Requires GROQ_API_KEY environment variable.
    Install: pip install groq
    """
    try:
        from groq import Groq
        client = Groq()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except ImportError:
        raise ImportError("Run:  pip install groq")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_question_paper(
    topic: str,
    difficulty: str = "medium",
    num_mcq: int = 5,
    num_short: int = 3,
    num_long: int = 2,
    rubric_criteria: list[str] | None = None,
    k: int = 8,
    vector_store: FAISS | None = None,
) -> str:
    """
    Main entry-point.

    Parameters
    ----------
    topic            : Subject / chapter the paper should cover.
    difficulty       : "easy" | "medium" | "hard"
    num_mcq          : Number of multiple-choice questions.
    num_short        : Number of short-answer questions.
    num_long         : Number of long/essay questions.
    rubric_criteria  : List of rubric criterion names from ALL_RUBRIC_CRITERIA.
                       Defaults to DEFAULT_RUBRIC if None.
    k                : Number of context chunks to retrieve from the vector store.
    vector_store     : Pre-loaded FAISS store (optional — loaded from disk if None).

    Returns
    -------
    str : The formatted question paper.
    """
    # 1. Load index
    if vector_store is None:
        print("Loading vector store …")
        vector_store = load_vector_store()

    # 2. Retrieve relevant chunks using balanced retrieval
    print(f"Retrieving top-{k} chunks for topic: '{topic}' using balanced retrieval …")
    chunks = balanced_retrieve(
        vector_store=vector_store,
        query=topic,
        k=k,
        difficulty=difficulty,
        diversity_weight=0.7,  # Higher weight for multi-topic relevance
        enable_difficulty_balance=False  # Keep topic balance priority
    )

    if not chunks:
        return (
            "⚠ No relevant content found in the uploaded material for this topic. "
            "Please upload course files (PDF/PPTX/DOCX) and rebuild the index."
        )

    # 3. Build context string (include source metadata)
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        src = chunk.metadata.get("source", "unknown")
        context_parts.append(f"[{i}] (from: {src})\n{chunk.page_content}")
    context = "\n\n".join(context_parts)

    # 4. Build prompt & call LLM
    prompt = _build_prompt(
        topic, context,
        num_mcq, num_short, num_long,
        difficulty, rubric_criteria
    )
    print("Generating question paper via LLM …")
    return _call_llm(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    paper = generate_question_paper(
        topic="Binary Search Trees",
        difficulty="hard",
        rubric_criteria=[
            "Conceptual clarity",
            "Application ability",
            "Time/space complexity",
            "Algorithm tracing",
            "Algorithm design",
        ],
    )
    print("\n" + paper)