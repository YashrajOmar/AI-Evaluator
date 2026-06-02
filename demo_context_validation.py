"""Test Step 5 (Context Injection) and Step 6 (Topic Distribution Validation)"""
from balanced_retrieval import (
    balanced_retrieve, 
    inject_context, 
    validate_topic_distribution,
    extract_topics
)
from vector_store import load_vector_store

print("Loading vector store...")
vs = load_vector_store()

query = "Normalization and Transactions"  # Change this to your DBMS topics
k = 8

print(f"\n{'='*70}")
print(f"Query: {query}")
print(f"{'='*70}\n")

# Retrieve chunks
print("Retrieving balanced chunks...")
chunks = balanced_retrieve(vs, query, k=k)

# STEP 5: Context Injection
print(f"\n{'='*70}")
print("STEP 5: CONTEXT INJECTION")
print(f"{'='*70}")

context = inject_context(chunks, include_metadata=True)
print(f"\n✓ Context created: {len(context)} characters")
print(f"\nPreview (first 400 chars):")
print("-" * 70)
print(context[:400])
print("...")
print("-" * 70)

# STEP 6: Topic Distribution Validation
print(f"\n{'='*70}")
print("STEP 6: TOPIC DISTRIBUTION VALIDATION")
print(f"{'='*70}")

topics = extract_topics(query)
is_valid, distribution, warnings = validate_topic_distribution(
    chunks, topics, min_coverage=0.25
)

print(f"\n{'='*70}")
print("SUMMARY:")
print(f"{'='*70}")
print(f"✓ Step 5 (Context Injection): IMPLEMENTED")
print(f"✓ Step 6 (Validation): IMPLEMENTED")
print(f"\nValidation Result: {'PASSED ✓' if is_valid else 'FAILED ✗'}")
print(f"Distribution: {distribution}")
if warnings:
    print(f"Warnings: {len(warnings)}")
print(f"{'='*70}")
