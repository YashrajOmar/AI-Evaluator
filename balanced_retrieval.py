"""
balanced_retrieval.py
Implements balanced retrieval to ensure diverse topic coverage when multiple topics are queried.

Problem: Standard similarity search focuses on the most similar topic, ignoring others.
Solution: Multi-topic balancing + MMR for diversity + difficulty alignment.
"""

from __future__ import annotations
import numpy as np
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# Topic Detection & Balancing
# ─────────────────────────────────────────────────────────────────────────────

def extract_topics(query: str) -> List[str]:
    """
    Extract individual topics from a multi-topic query.
    
    Examples:
        "BST and AVL Trees" → ["BST", "AVL Trees"]
        "Sorting, Searching and Hashing" → ["Sorting", "Searching", "Hashing"]
    """
    # Split on common separators (order matters - check longer patterns first)
    separators = [' and ', ', ', ' & ']
    topics = [query]
    
    for sep in separators:
        new_topics = []
        for topic in topics:
            if sep in topic:
                new_topics.extend([t.strip() for t in topic.split(sep)])
            else:
                new_topics.append(topic)
        topics = new_topics
    
    # Final cleanup - split on comma if still present
    final_topics = []
    for topic in topics:
        if ',' in topic:
            final_topics.extend([t.strip() for t in topic.split(',')])
        else:
            final_topics.append(topic)
    
    return [t for t in final_topics if t and len(t) > 1]  # Remove empty/single char strings


def retrieve_per_topic(
    vector_store: FAISS,
    topics: List[str],
    k_per_topic: int
) -> List[Document]:
    """
    Retrieve k chunks for each topic separately.
    Ensures all topics get representation.
    """
    all_chunks = []
    seen_content = set()  # Avoid duplicates
    
    for topic in topics:
        print(f"  Retrieving for topic: '{topic}'")
        chunks = vector_store.similarity_search(topic, k=k_per_topic)
        
        added = 0
        for chunk in chunks:
            # Use first 200 chars as fingerprint to detect duplicates
            fingerprint = chunk.page_content[:200].strip()
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                all_chunks.append(chunk)
                added += 1
        
        print(f"    Added {added} unique chunks for '{topic}'")
    
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# MMR (Maximal Marginal Relevance) for Diversity
# ─────────────────────────────────────────────────────────────────────────────

def mmr_rerank(
    query: str,
    chunks: List[Document],
    vector_store: FAISS,
    k: int,
    lambda_param: float = 0.5
) -> List[Document]:
    """
    Re-rank chunks using MMR to maximize diversity.
    
    MMR = λ * Similarity(query, chunk) - (1-λ) * max(Similarity(chunk, selected))
    
    Parameters:
        lambda_param: 0 = max diversity, 1 = max relevance (default: 0.5 for balance)
    """
    if len(chunks) <= k:
        return chunks
    
    embeddings = vector_store.embeddings
    
    # Get embeddings
    query_embedding = embeddings.embed_query(query)
    chunk_embeddings = [embeddings.embed_query(c.page_content) for c in chunks]
    
    # Convert to numpy arrays
    query_vec = np.array(query_embedding)
    chunk_vecs = np.array(chunk_embeddings)
    
    # Calculate query similarities
    query_sims = cosine_similarity_batch(query_vec, chunk_vecs)
    
    selected_indices = []
    remaining_indices = list(range(len(chunks)))
    
    # Select first chunk (highest similarity)
    first_idx = int(np.argmax(query_sims))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Select remaining k-1 chunks using MMR
    for _ in range(k - 1):
        if not remaining_indices:
            break
        
        mmr_scores = []
        selected_vecs = chunk_vecs[selected_indices]
        
        for idx in remaining_indices:
            # Relevance to query
            relevance = query_sims[idx]
            
            # Max similarity to already selected chunks
            similarities = cosine_similarity_batch(chunk_vecs[idx], selected_vecs)
            max_sim = np.max(similarities) if len(similarities) > 0 else 0
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append(mmr)
        
        # Select chunk with highest MMR
        best_idx = remaining_indices[int(np.argmax(mmr_scores))]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return [chunks[i] for i in selected_indices]


def cosine_similarity_batch(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between a vector and multiple vectors."""
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    
    dot_products = np.dot(matrix, vec)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
    
    # Avoid division by zero
    norms = np.where(norms == 0, 1e-10, norms)
    
    return dot_products / norms


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty Estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_difficulty(chunk: Document) -> str:
    """
    Estimate difficulty level of a chunk based on heuristics.
    Returns: "easy", "medium", or "hard"
    """
    text = chunk.page_content.lower()
    
    # Hard indicators
    hard_keywords = [
        'prove', 'proof', 'theorem', 'lemma', 'complexity analysis',
        'design algorithm', 'optimize', 'trade-off', 'amortized',
        'correctness', 'invariant', 'recurrence'
    ]
    
    # Easy indicators
    easy_keywords = [
        'definition', 'define', 'what is', 'introduction', 'basic',
        'simple', 'example:', 'for example', 'overview'
    ]
    
    hard_count = sum(1 for kw in hard_keywords if kw in text)
    easy_count = sum(1 for kw in easy_keywords if kw in text)
    
    # Length heuristic (longer chunks tend to be more complex)
    length = len(text)
    
    if hard_count >= 2 or length > 1000:
        return "hard"
    elif easy_count >= 2 or length < 300:
        return "easy"
    else:
        return "medium"


def balance_by_difficulty(
    chunks: List[Document],
    target_difficulty: str,
    k: int
) -> List[Document]:
    """
    Balance chunks according to target difficulty while preserving topic diversity.
    
    Distribution:
        easy:   60% easy, 30% medium, 10% hard
        medium: 20% easy, 60% medium, 20% hard
        hard:   10% easy, 30% medium, 60% hard
    """
    if len(chunks) <= k:
        return chunks
    
    # Classify chunks
    classified = {
        'easy': [],
        'medium': [],
        'hard': []
    }
    
    for chunk in chunks:
        diff = estimate_difficulty(chunk)
        classified[diff].append(chunk)
    
    # Define target ratios
    ratios = {
        'easy':   {'easy': 0.6, 'medium': 0.3, 'hard': 0.1},
        'medium': {'easy': 0.2, 'medium': 0.6, 'hard': 0.2},
        'hard':   {'easy': 0.1, 'medium': 0.3, 'hard': 0.6},
    }
    
    target_ratio = ratios.get(target_difficulty, ratios['medium'])
    
    # Calculate how many chunks to take from each difficulty
    k_easy = max(1, int(k * target_ratio['easy']))
    k_medium = max(1, int(k * target_ratio['medium']))
    k_hard = k - k_easy - k_medium  # Remaining
    
    # Select chunks (round-robin to preserve order/diversity from MMR)
    selected = []
    
    # Take proportionally from each category
    easy_chunks = classified['easy'][:k_easy] if classified['easy'] else []
    medium_chunks = classified['medium'][:k_medium] if classified['medium'] else []
    hard_chunks = classified['hard'][:k_hard] if classified['hard'] else []
    
    # Interleave to maintain diversity
    all_selected = easy_chunks + medium_chunks + hard_chunks
    
    # If we don't have enough, fill from remaining chunks in order
    if len(all_selected) < k:
        remaining = [c for c in chunks if c not in all_selected]
        all_selected.extend(remaining[:k - len(all_selected)])
    
    return all_selected[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Main Balanced Retrieval Function
# ─────────────────────────────────────────────────────────────────────────────

def balanced_retrieve(
    vector_store: FAISS,
    query: str,
    k: int = 8,
    difficulty: str = "medium",
    diversity_weight: float = 0.5,
    enable_difficulty_balance: bool = False  # Disabled by default to preserve topic balance
) -> List[Document]:
    """
    Balanced retrieval with multi-topic coverage, diversity, and difficulty alignment.
    
    Parameters:
        vector_store: FAISS vector store
        query: Search query (can contain multiple topics)
        k: Number of chunks to retrieve
        difficulty: Target difficulty level ("easy", "medium", "hard")
        diversity_weight: MMR lambda parameter (0=max diversity, 1=max relevance)
        enable_difficulty_balance: Whether to balance by difficulty (may affect topic balance)
    
    Returns:
        List of balanced, diverse chunks covering all topics
    """
    print(f"\n[Balanced Retrieval] Query: '{query}'")
    
    # Step 1: Detect multiple topics
    topics = extract_topics(query)
    
    if len(topics) > 1:
        print(f"[Step 1] Detected {len(topics)} topics: {topics}")
        
        # Retrieve equal chunks per topic and interleave them
        k_per_topic = (k // len(topics)) + 2  # Get a bit extra per topic
        
        topic_chunks = {}
        for topic in topics:
            print(f"  Retrieving for topic: '{topic}'")
            chunks = vector_store.similarity_search(topic, k=k_per_topic)
            topic_chunks[topic] = chunks
            print(f"    Retrieved {len(chunks)} chunks for '{topic}'")
        
        # Interleave chunks from each topic (round-robin)
        candidate_chunks = []
        seen_content = set()
        max_len = max(len(chunks) for chunks in topic_chunks.values())
        
        for i in range(max_len):
            for topic in topics:
                if i < len(topic_chunks[topic]):
                    chunk = topic_chunks[topic][i]
                    fingerprint = chunk.page_content[:200].strip()
                    if fingerprint not in seen_content:
                        seen_content.add(fingerprint)
                        candidate_chunks.append(chunk)
        
        print(f"[Step 2] Interleaved {len(candidate_chunks)} unique chunks from all topics")
        
    else:
        print(f"[Step 1] Single topic detected: '{topics[0]}'")
        # Single topic: retrieve more candidates for MMR
        candidate_chunks = vector_store.similarity_search(query, k=k*3)
        print(f"[Step 2] Retrieved {len(candidate_chunks)} candidate chunks")
    
    if not candidate_chunks:
        print("[Warning] No chunks found!")
        return []
    
    # Step 2: Apply MMR for diversity (but not too aggressive for multi-topic)
    if len(topics) > 1:
        # For multi-topic, use higher lambda to preserve relevance
        mmr_lambda = max(0.7, diversity_weight)
    else:
        mmr_lambda = diversity_weight
    
    num_for_mmr = min(k*2, len(candidate_chunks))
    diverse_chunks = mmr_rerank(
        query,
        candidate_chunks,
        vector_store,
        k=num_for_mmr,
        lambda_param=mmr_lambda
    )
    
    print(f"[Step 3] After MMR (lambda={mmr_lambda}): {len(diverse_chunks)} diverse chunks")
    
    # Step 3: Select final k chunks
    if enable_difficulty_balance and len(diverse_chunks) > k:
        final_chunks = balance_by_difficulty(diverse_chunks, difficulty, k)
        print(f"[Step 4] After difficulty balancing: {len(final_chunks)} chunks")
    else:
        # Just take top k from MMR (preserves topic diversity better)
        final_chunks = diverse_chunks[:k]
        print(f"[Step 4] Selected top {len(final_chunks)} chunks (preserving topic diversity)")
    
    return final_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Context Injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_context(chunks: List[Document], include_metadata: bool = True) -> str:
    """
    Convert retrieved chunks into formatted context string for LLM.
    
    Parameters:
        chunks: List of retrieved document chunks
        include_metadata: Whether to include source metadata
    
    Returns:
        Formatted context string ready for LLM prompt
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        if include_metadata:
            src = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "")
            page_info = f" [Page {page}]" if page else ""
            context_parts.append(f"[{i}] (from: {src}{page_info})\n{chunk.page_content}")
        else:
            context_parts.append(f"[{i}]\n{chunk.page_content}")
    
    context = "\n\n".join(context_parts)
    return context


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Topic Distribution Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_topic_distribution(
    chunks: List[Document],
    topics: List[str],
    min_coverage: float = 0.25
) -> tuple[bool, dict, list]:
    """
    Validate that all topics are adequately represented in retrieved chunks.
    
    Parameters:
        chunks: Retrieved document chunks
        topics: List of topics that should be covered
        min_coverage: Minimum fraction of chunks per topic (default: 0.25 = 25%)
    
    Returns:
        Tuple of (is_valid, distribution, warnings)
        - is_valid: True if all topics meet minimum coverage
        - distribution: Dict mapping topic to chunk count
        - warnings: List of warning messages for underrepresented topics
    """
    if not chunks or not topics:
        return False, {}, ["No chunks or topics provided"]
    
    # Count chunks per topic
    distribution = {}
    chunk_topic_map = {}  # Track which chunks belong to which topics
    
    for topic in topics:
        topic_lower = topic.lower()
        count = 0
        chunk_indices = []
        
        for idx, chunk in enumerate(chunks):
            content_lower = chunk.page_content.lower()
            
            # Check if topic keywords are in chunk
            # Handle multi-word topics (e.g., "AVL Trees")
            topic_words = topic_lower.split()
            if all(word in content_lower for word in topic_words):
                count += 1
                chunk_indices.append(idx)
        
        distribution[topic] = count
        chunk_topic_map[topic] = chunk_indices
    
    # Validate coverage
    total = len(chunks)
    warnings = []
    
    for topic, count in distribution.items():
        coverage = count / total if total > 0 else 0
        
        if coverage < min_coverage:
            warnings.append(
                f"⚠ Topic '{topic}' is underrepresented: "
                f"{count}/{total} chunks ({coverage:.1%}) - "
                f"minimum required: {min_coverage:.1%}"
            )
        elif coverage == 0:
            warnings.append(
                f"❌ Topic '{topic}' has NO chunks! "
                f"Consider adding more documents or adjusting query."
            )
    
    is_valid = len(warnings) == 0
    
    # Print validation summary
    print(f"\n[Topic Distribution Validation]")
    print(f"Total chunks: {total}")
    for topic, count in distribution.items():
        coverage = count / total if total > 0 else 0
        status = "✓" if coverage >= min_coverage else "✗"
        print(f"  {status} {topic}: {count} chunks ({coverage:.1%})")
    
    if warnings:
        print(f"\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print(f"\n✓ All topics adequately represented!")
    
    return is_valid, distribution, warnings


def balanced_retrieve_with_validation(
    vector_store: FAISS,
    query: str,
    k: int = 8,
    difficulty: str = "medium",
    min_coverage: float = 0.25,
    max_retries: int = 2
) -> tuple[List[Document], dict]:
    """
    Balanced retrieval with automatic validation and retry.
    
    If initial retrieval doesn't meet coverage requirements, 
    automatically retries with adjusted parameters.
    
    Returns:
        Tuple of (chunks, metadata)
        - chunks: Retrieved document chunks
        - metadata: Dict with validation info and stats
    """
    attempt = 0
    best_chunks = None
    best_distribution = None
    
    while attempt < max_retries:
        attempt += 1
        print(f"\n{'='*70}")
        print(f"Retrieval Attempt {attempt}/{max_retries}")
        print(f"{'='*70}")
        
        # Adjust k based on attempt (get more chunks on retry)
        adjusted_k = k + (attempt - 1) * 2
        
        # Retrieve chunks
        chunks = balanced_retrieve(
            vector_store, query, k=adjusted_k, 
            difficulty=difficulty, diversity_weight=0.5
        )
        
        if not chunks:
            print("No chunks retrieved!")
            continue
        
        # Validate distribution
        topics = extract_topics(query)
        is_valid, distribution, warnings = validate_topic_distribution(
            chunks, topics, min_coverage
        )
        
        # Keep best result
        if best_chunks is None or is_valid:
            best_chunks = chunks[:k]  # Return only k chunks
            best_distribution = distribution
        
        if is_valid:
            print(f"\n✓ Validation passed on attempt {attempt}!")
            break
        else:
            print(f"\n✗ Validation failed. Retrying with k={adjusted_k+2}...")
    
    # Prepare metadata
    metadata = {
        "attempts": attempt,
        "distribution": best_distribution,
        "is_valid": is_valid if 'is_valid' in locals() else False,
        "warnings": warnings if 'warnings' in locals() else []
    }
    
    return best_chunks, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Testing & Comparison
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from vector_store import load_vector_store, retrieve_chunks
    
    print("Loading vector store...")
    vs = load_vector_store()
    
    # Test queries
    test_queries = [
        ("Sorting and AVL Trees", 8),
        ("Heap Sort and Quick Sort", 6),
    ]
    
    for query, k in test_queries:
        print(f"\n{'='*70}")
        print(f"TEST: {query}")
        print(f"{'='*70}\n")
        
        # Test Step 5: Context Injection
        print("STEP 5: Context Injection")
        print("-" * 70)
        chunks = balanced_retrieve(vs, query, k=k)
        context = inject_context(chunks, include_metadata=True)
        print(f"Context length: {len(context)} characters")
        print(f"Preview:\n{context[:300]}...\n")
        
        # Test Step 6: Topic Distribution Validation
        print("\nSTEP 6: Topic Distribution Validation")
        print("-" * 70)
        topics = extract_topics(query)
        is_valid, distribution, warnings = validate_topic_distribution(
            chunks, topics, min_coverage=0.25
        )
        
        # Test with validation and retry
        print(f"\n{'='*70}")
        print("BALANCED RETRIEVE WITH AUTO-VALIDATION:")
        print("=" * 70)
        final_chunks, metadata = balanced_retrieve_with_validation(
            vs, query, k=k, min_coverage=0.30, max_retries=2
        )
        
        print(f"\nFinal Result:")
        print(f"  Chunks: {len(final_chunks)}")
        print(f"  Attempts: {metadata['attempts']}")
        print(f"  Valid: {metadata['is_valid']}")
        print(f"  Distribution: {metadata['distribution']}")
        
        print(f"\n{'='*70}\n")
