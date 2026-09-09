"""
rebuild.py - Transactional Rebuild & Maintenance CLI for Cortex Personal Brain.

Commands:
- python rebuild.py --embeddings   : Drops and re-embeds ChromaDB from all verified raw documents.
- python rebuild.py --graph        : Re-extracts semantic triples, constructs the Hierarchical Graph, and computes Louvain communities.
- python rebuild.py --all          : Complete clean wipe and rebuild of both Vector Store and Hierarchical Graph.
- python rebuild.py --status       : Inspects current chunk counts, graph nodes, and community summaries.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

from tqdm import tqdm

from config import SourceTier, settings
from ingest import DataIngestionPipeline, ProcessedChunk, ChromaStore, LocalEmbeddingEngine
from graph_engine import HierarchicalTemporalGraph, SemanticTriple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cortex.rebuild")


def get_raw_txt_files(raw_dir: Path = settings.raw_data_dir) -> List[Path]:
    """Returns sorted list of all active verified text documents on disk."""
    files = [f for f in raw_dir.glob("*.txt") if not f.name.endswith(".meta.json")]
    return sorted(files)


def rebuild_embeddings(raw_dir: Path = settings.raw_data_dir, limit: int = None):
    """Drops ChromaDB and re-chunks and re-embeds every verified raw document on disk."""
    files = get_raw_txt_files(raw_dir)
    if limit:
        files = files[:limit]

    logger.info(f"Starting Vector Store Rebuild across {len(files)} files...")
    ingestor = DataIngestionPipeline()
    ingestor.chroma_store.clear()

    total_chunks = 0
    pbar = tqdm(files, desc="[Rebuilding Embeddings]", unit="file")
    for f in pbar:
        try:
            chunks = ingestor.ingest_file(f)
            total_chunks += len(chunks)
            pbar.set_postfix({"total_indexed_chunks": total_chunks})
        except Exception as e:
            logger.warning(f"Error ingesting {f.name}: {e}")

    print(f"\n[DONE] Vector Store Rebuild Complete! Indexed {total_chunks} deduplicated chunks into ChromaDB.")
    return total_chunks


def rebuild_graph(raw_dir: Path = settings.raw_data_dir, limit: int = None, use_llm: bool = False):
    """Reconstructs the Hierarchical Temporal Graph and detects Louvain communities."""
    files = get_raw_txt_files(raw_dir)
    if limit:
        files = files[:limit]

    logger.info(f"Starting Hierarchical Graph Rebuild across {len(files)} files (use_llm={use_llm})...")
    graph = HierarchicalTemporalGraph()
    graph.clear()

    ingestor = DataIngestionPipeline()
    total_triples = 0
    total_chunks_processed = 0

    pbar = tqdm(files, desc="[Rebuilding Epistemic Graph]", unit="file")
    for f in pbar:
        try:
            # Chunk file without re-embedding
            with open(f, "r", encoding="utf-8", errors="ignore") as tf:
                text = tf.read()
            tier = ingestor.doc_processor.infer_tier(f.name, text)
            year = ingestor.doc_processor.extract_year_from_text(text)
            chunks = ingestor.chunk_text(text, f.stem, f.name, tier, year)

            for chunk in chunks:
                triples = graph.index_chunk(chunk, use_llm=use_llm)
                total_triples += len(triples)
                total_chunks_processed += 1

            pbar.set_postfix({"nodes": graph.graph.number_of_nodes(), "triples": total_triples})
        except Exception as e:
            logger.warning(f"Error processing graph for {f.name}: {e}")

    graph.save()
    logger.info(f"Graph Construction Complete: {graph.graph.number_of_nodes()} nodes, {graph.graph.number_of_edges()} edges.")

    # Detect communities and compute macro summaries
    logger.info("Computing Louvain Community Clusters...")
    communities = graph.detect_and_build_communities()
    print(f"\n[DONE] Hierarchical Graph Rebuild Complete! Generated {len(communities)} thematic communities and {total_triples} epistemic triples.")
    return communities


def show_status():
    """Prints current database and graph status."""
    chroma_store = ChromaStore(settings.chroma_db_dir)
    chunks = chroma_store.load_all_chunks()
    graph = HierarchicalTemporalGraph()

    print("\n" + "=" * 60)
    print("=== CORTEX PERSONAL BRAIN DATABASE & GRAPH STATUS ===")
    print("=" * 60)
    print(f"Raw Verified Documents on Disk: {len(get_raw_txt_files())}")
    print(f"ChromaDB Indexed Chunks:        {len(chunks)}")
    print(f"Knowledge Graph Nodes:          {graph.graph.number_of_nodes()}")
    print(f"Knowledge Graph Edges:          {graph.graph.number_of_edges()}")
    print(f"Hierarchical Communities:       {len(graph.communities)}")
    if graph.communities:
        print("\nThematic Communities Detected:")
        for cid, comm in list(graph.communities.items())[:5]:
            print(f" - [{comm.year_range[0]}-{comm.year_range[1]}] {comm.title}")
    print("=" * 60 + "\n")


def add_custom_file(file_path: Path, tier: int = 1, year: int = 2024, title: Optional[str] = None):
    """
    Ingests and indexes a single custom file (.txt, .pdf, .docx, .md, audio)
    with custom tier and year metadata into ChromaDB and Hierarchical Graph.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Copy to raw data dir if not already there
    dest_path = settings.raw_data_dir / file_path.name
    if file_path.resolve() != dest_path.resolve():
        import shutil
        shutil.copyfile(file_path, dest_path)
        logger.info(f"Copied {file_path.name} to {dest_path}")

    # Write companion metadata
    meta_path = dest_path.with_suffix(".meta.json")
    meta_data = {
        "tier": tier,
        "year": year,
        "source_name": title or file_path.stem.replace("_", " ").title(),
        "source_type": "custom_upload"
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta_data, mf, indent=2)

    logger.info(f"Indexing custom document: {dest_path.name} (Tier {tier}, Year {year})...")
    from pipeline import CortexPersonalBrain
    brain = CortexPersonalBrain()
    chunks = brain.ingest_document(dest_path, forced_tier=SourceTier(tier), forced_year=year)
    print(f"\n[DONE] Successfully ingested and indexed {len(chunks)} chunks from '{dest_path.name}' into ChromaDB & Knowledge Graph!")
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Transactional Rebuild CLI for Cortex Knowledge Graph & Embeddings")
    parser.add_argument("--embeddings", action="store_true", help="Drop and rebuild ChromaDB vector embeddings")
    parser.add_argument("--graph", action="store_true", help="Re-extract epistemic triples and rebuild Hierarchical Graph & Communities")
    parser.add_argument("--all", action="store_true", help="Wipe and rebuild both Vector Store and Hierarchical Graph")
    parser.add_argument("--use-llm", action="store_true", help="Use local Qwen 27B LLM for deep triple extraction (slower)")
    parser.add_argument("--status", action="store_true", help="Display current status of database, embeddings, and graph")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of raw documents to process (default: all)")
    parser.add_argument("--add-file", type=str, help="Path to a custom file (.txt, .pdf, .docx, .md, .mp3) to ingest")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3, 4], help="Source tier for custom file: 1 (Oral Interview), 2 (Lecture), 3 (Paper), 4 (Third-Party)")
    parser.add_argument("--year", type=int, default=2024, help="Year of the custom file (default: 2024)")
    parser.add_argument("--title", type=str, default=None, help="Descriptive title for the custom file")
    args = parser.parse_args()

    if args.add_file:
        add_custom_file(Path(args.add_file), tier=args.tier, year=args.year, title=args.title)
        show_status()
        return

    if args.status:
        show_status()
        return

    if args.all:
        logger.info("Starting Full Rebuild (Vector Store + Hierarchical Graph)...")
        rebuild_embeddings(limit=args.limit)
        rebuild_graph(limit=args.limit, use_llm=args.use_llm)
        show_status()
        return

    if args.embeddings:
        rebuild_embeddings(limit=args.limit)
        show_status()
        return

    if args.graph:
        rebuild_graph(limit=args.limit, use_llm=args.use_llm)
        show_status()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
