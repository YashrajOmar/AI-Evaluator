"""
main.py
CLI for the AI Evaluator — RAG-powered question paper generator.

Usage
-----
# Step 1 — Professor uploads files to data/ folder, then rebuild index:
    python main.py --build

# Step 2 — Generate a question paper:
    python main.py --topic "Binary Search Trees" --difficulty hard

# Step 3 — Save output to a file:
    python main.py --topic "Sorting Algorithms" --output paper.txt
"""
from dotenv import load_dotenv
load_dotenv()
import argparse
import sys
from pathlib import Path


def build_index(source_dir: str, multimodal: bool = False) -> None:
    if multimodal:
        from multimodal_ingestion import build_multimodal_vector_store
        print(f"\n{'='*50}")
        print("Building / Rebuilding Multimodal Vector Index")
        print("(Text + Images)")
        print(f"{'='*50}\n")
        build_multimodal_vector_store(source_dir, extract_images=True)
    else:
        from vector_store import build_vector_store
        print(f"\n{'='*50}")
        print("Building / Rebuilding Vector Index")
        print(f"{'='*50}\n")
        build_vector_store(source_dir)
    
    print("\n✓ Index ready. You can now generate question papers.\n")


def generate(topic: str, difficulty: str, num_mcq: int, num_short: int,
             num_long: int, output: str | None, rubric_criteria: list = None) -> None:
    from rag_pipeline import generate_question_paper

    print(f"\n{'='*50}")
    print(f"Generating Question Paper")
    print(f"Topic: {topic} | Difficulty: {difficulty}")
    print(f"{'='*50}\n")

    paper = generate_question_paper(
        topic=topic,
        difficulty=difficulty,
        num_mcq=num_mcq,
        num_short=num_short,
        num_long=num_long,
        rubric_criteria=rubric_criteria,
    )

    print("\n" + paper)

    if output:
        Path(output).write_text(paper, encoding="utf-8")
        print(f"\n✓ Paper saved to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Evaluator — RAG-powered question paper generator"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="(Re)build the FAISS index from files in --data-dir",
    )
    parser.add_argument(
        "--multimodal", action="store_true",
        help="Enable multimodal mode (extract and caption images from PDFs)",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Folder containing PDFs, PPTXs, DOCXs, TXTs (default: data/)",
    )
    parser.add_argument("--topic",      default="Data Structures and Algorithms")
    parser.add_argument("--difficulty", default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--num-mcq",   type=int, default=5)
    parser.add_argument("--num-short", type=int, default=3)
    parser.add_argument("--num-long",  type=int, default=2)
    parser.add_argument("--output",    default=None,
                        help="Optional .txt file to save the paper")
    parser.add_argument("--rubric", nargs="+", 
    default=["Conceptual clarity", "Application ability", "Time/space complexity"])

    args = parser.parse_args()

    if args.build:
        build_index(args.data_dir, args.multimodal)
        return

    generate(
        topic=args.topic,
        difficulty=args.difficulty,
        num_mcq=args.num_mcq,
        num_short=args.num_short,
        num_long=args.num_long,
        output=args.output,
        rubric_criteria=args.rubric,
    )


if __name__ == "__main__":
    main()