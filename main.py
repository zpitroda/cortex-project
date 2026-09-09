"""
main.py - CLI & Benchmark Evaluation Harness for Cortex Personal Brain.

Usage:
    python main.py --ingest-all
    python main.py --query "Why did I move from Drosophila to yeast?"
    python main.py --benchmark
"""

import sys
import json
import argparse
import logging
from pathlib import Path

from config import settings
from pipeline import CortexPersonalBrain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cortex.main")

BENCHMARK_QUESTIONS = [
    "Why did I leave Drosophila behind after graduate school, and what was I looking for in what came next?",
    "What did I learn from Norman Davidson that I didn't learn from Ron Davis, and the other way round?",
    "I turned down Harvard. What was the reason?",
    "How do I think about the politics of science, and about people who work the politics rather than the problem?",
    "What happened when I first went looking for autoimmune sera, and what did that episode teach me?",
    "Am I happier at the bench or running things? Has that changed?",
    "Which of my own papers did I consider the most important, and has my answer to that shifted over time?",
    "What did my high school chemistry class have to do with the way I run a lab?",
    "Why did I leave Yale for Stanford?",
    "When I decided to profile a human longitudinally, why did I start with myself rather than a cohort?",
    "What have I been wrong about? Name something I believed early that the field or the data later contradicted.",
    "How did my view on competition, and on working in crowded areas, hold up over my career?",
]


def print_formatted_result(res: dict):
    """Renders a clean human-readable output of the Brain query result."""
    print("=" * 80)
    print(f"QUESTION: {res['question']}")
    print("=" * 80)
    print("\n[NARRATIVE - EVOLUTION OF REASONING]")
    print(res["narrative"])
    print("\n[TIMELINE SHIFTS]")
    for shift in res.get("timeline_shifts", []):
        print(f"  * Year {shift['year']} [Chunk: {shift['chunk_id'][:8]}...]: {shift['belief']}")
    print("\n[CITATIONS & PROVENANCE]")
    for cite in res.get("citations", []):
        print(f"  * [Chunk: {cite['chunk_id'][:8]}...]: \"{cite['exact_quote']}\"")
    print("\n[FLAGS]")
    print(f"  * Inferred: {res['flags']['is_inferred']} | Thin Record: {res['flags']['thin_record']}")
    
    conf = res["confidence"]
    print("\n" + "-" * 50)
    print(f"MATHEMATICAL CONFIDENCE SCORE: {conf['final_score']:.1f}%")
    print(f"  * Base Score (Tier):       +{conf['base_score']:.1f} / 35.0")
    print(f"  * Corroboration (Years):   +{conf['corroboration_score']:.1f} / 30.0 (Years: {conf['cited_years']})")
    print(f"  * Recency Score:           +{conf.get('recency_score', 0.0):.1f} / 10.0")
    print(f"  * Quote Verification:      +{conf['verification_score']:.1f} / 25.0")
    print(f"  * Penalties Deducted:      -{conf['penalty_score']:.1f}")
    print(f"  * Rationale: {conf['mathematical_rationale']}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Cortex Personal Brain for Michael Snyder")
    parser.add_argument("--ingest-all", action="store_true", help="Ingest all raw documents in l:/cortex")
    parser.add_argument("--query", type=str, help="Query Michael Snyder's personal brain")
    parser.add_argument("--benchmark", action="store_true", help="Run the 12 benchmark historical questions")
    parser.add_argument("--output-json", type=str, default="", help="Save benchmark results to JSON file")
    parser.add_argument("--serve", action="store_true", help="Start the interactive Web Dashboard & REST API server")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        from server import app
        print("=" * 80)
        print("=== STARTING CORTEX PERSONAL BRAIN INTERACTIVE WEB SERVER ===")
        print("=== Open your browser at: http://127.0.0.1:8000           ===")
        print("=" * 80)
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
        return

    brain = CortexPersonalBrain()

    if args.ingest_all:
        logger.info("Ingesting all documents from corpus...")
        count = brain.ingest_directory(settings.base_dir)
        print(f"\nIngestion complete! Successfully indexed {count} unique semantic chunks.")

    if args.query:
        res = brain.query(args.query)
        print_formatted_result(res)

    if args.benchmark:
        print("\n=== RUNNING 12 BENCHMARK HISTORICAL QUESTIONS ===\n")
        
        all_results = []
        md_lines = ["# Cortex Personal Brain: 12 Benchmark Questions Transcript\n\n"]

        for i, q in enumerate(BENCHMARK_QUESTIONS, 1):
            print(f"\n>>> Benchmark [{i}/12]: {q}")
            res = brain.query(q)
            print_formatted_result(res)
            all_results.append(res)

            conf = res["confidence"]
            md_lines.append(f"## [{i}/12] Question: {q}\n")
            md_lines.append(f"**Mathematical Confidence**: `{conf['final_score']:.1f}%` (Base: +{conf['base_score']:.1f}, Corroboration: +{conf['corroboration_score']:.1f}, Recency: +{conf['recency_score']:.1f}, Verification: +{conf['verification_score']:.1f}, Penalties: -{conf['penalty_score']:.1f})\n\n")
            md_lines.append(f"### Narrative\n{res['narrative']}\n\n")
            md_lines.append("### Timeline Shifts\n")
            for s in res.get("timeline_shifts", []):
                md_lines.append(f"- **Year {s['year']}** (`{s['chunk_id'][:8]}...`): {s['belief']}\n")
            md_lines.append("\n### Citations & Provenance\n")
            for c in res.get("citations", []):
                md_lines.append(f"- `[{c['chunk_id'][:8]}...]`: *\"{c['exact_quote']}\"*\n")
            md_lines.append("\n---\n\n")

        json_out = Path(args.output_json or "L:/cortex/benchmark_transcript.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved benchmark JSON transcript to {json_out}")

        md_out = Path("L:/cortex/benchmark_transcript.md")
        with open(md_out, "w", encoding="utf-8") as f:
            f.writelines(md_lines)
        print(f"Saved benchmark Markdown transcript to {md_out}")


if __name__ == "__main__":
    main()
