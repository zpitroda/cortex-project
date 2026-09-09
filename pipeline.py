"""
pipeline.py - Central Orchestrator for Cortex Personal Brain.

Coordinates:
1. Data Ingestion & 3-Step Deduplication (ingest.py)
2. Temporal Graph Construction & Anti-Flattening Stratified Retrieval (retrieval.py)
3. LLM Client with Truncation Bypass (llm_client.py)
4. Mathematical Confidence Evaluation (confidence.py)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import SourceTier, LLMResponseSchema, settings
from ingest import DataIngestionPipeline, ProcessedChunk, DocumentProcessor, extract_snyder_speech
from retrieval import TemporalGraphExtractor, AntiFlatteningRetriever
from graph_engine import HierarchicalTemporalGraph
from llm_client import CortexLLMClient
from confidence import MathematicalConfidenceEngine, ConfidenceReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cortex.pipeline")


class CortexPersonalBrain:
    """
    Main system interface for Michael Snyder's Personal Brain.
    """
    def __init__(self):
        self.ingestion = DataIngestionPipeline()
        self.graph_extractor = TemporalGraphExtractor()
        self.hierarchical_graph = HierarchicalTemporalGraph()
        self.retriever = AntiFlatteningRetriever(
            chroma_store=self.ingestion.chroma_store,
            embed_engine=self.ingestion.embed_engine,
            graph_extractor=self.graph_extractor,
            max_chunks=settings.retrieval.max_llm_chunks,
        )
        self.llm_client = CortexLLMClient()
        self.confidence_engine = MathematicalConfidenceEngine()
        self.chunk_database: Dict[str, Dict[str, Any]] = {}
        self._load_active_chunks()

    def _load_active_chunks(self):
        """Loads in-memory database of chunks from ChromaDB and active dedup engine for verification."""
        persisted_chunks = self.ingestion.chroma_store.load_all_chunks()
        for c in persisted_chunks:
            cid = c.get("chunk_id")
            if cid:
                meta = c.get("metadata", {})
                raw_text = c.get("raw_text", "")
                snyder_text = extract_snyder_speech(raw_text)
                self.chunk_database[cid] = {
                    "chunk_id": cid,
                    "source_id": meta.get("source_id", cid),
                    "source_name": meta.get("source_name", "Archived Source"),
                    "tier": int(meta.get("tier", 1)),
                    "year": int(meta.get("year", 1991)),
                    "raw_text": raw_text,
                    "snyder_text": snyder_text,
                }

        for chunk_id, chunk in self.ingestion.dedup_engine.active_chunks.items():
            self.chunk_database[chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "source_name": chunk.source_name,
                "tier": chunk.tier,
                "year": chunk.year,
                "raw_text": chunk.raw_text,
                "snyder_text": chunk.snyder_text,
            }

    def ingest_document(self, file_path: Path, forced_tier: Optional[SourceTier] = None, forced_year: Optional[int] = None) -> List[ProcessedChunk]:
        """Ingests a single file, builds graph relations, and indexes chunks."""
        chunks = self.ingestion.ingest_file(file_path, forced_tier, forced_year)
        for chunk in chunks:
            self.graph_extractor.index_chunk(chunk)
            self.hierarchical_graph.index_chunk(chunk)
            self.chunk_database[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "source_name": chunk.source_name,
                "tier": chunk.tier,
                "year": chunk.year,
                "raw_text": chunk.raw_text,
                "snyder_text": chunk.snyder_text,
            }
        self.graph_extractor.save_graph()
        self.hierarchical_graph.save()
        return chunks

    def ingest_directory(self, dir_path: Path) -> int:
        """Batch ingests all supported files in a directory."""
        dir_path = Path(dir_path)
        valid_extensions = {".docx", ".pdf", ".txt", ".md", ".mp3", ".wav", ".m4a"}
        total_chunks = 0
        
        for f in dir_path.iterdir():
            if f.is_file() and f.suffix.lower() in valid_extensions and not f.name.startswith("~"):
                try:
                    chunks = self.ingest_document(f)
                    total_chunks += len(chunks)
                except Exception as e:
                    logger.error(f"Error ingesting {f.name}: {e}")

        logger.info(f"Ingestion complete: {total_chunks} total unique chunks indexed.")
        return total_chunks

    def query(
        self,
        question: str,
        mock_response_generator: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end query answering:
        1. Stratified Anti-Flattening Retrieval (Max 15 chunks)
        2. LLM Call with Truncation Bypass
        3. Mathematical Confidence Calculation
        """
        logger.info(f"=== Processing Personal Brain Query: '{question}' ===")

        # 1. Retrieval
        retrieved_chunks = self.retriever.retrieve(question)

        # Register retrieved chunks in chunk database if not present
        for c in retrieved_chunks:
            cid = c.get("chunk_id")
            if cid and cid not in self.chunk_database:
                meta = c.get("metadata", {})
                self.chunk_database[cid] = {
                    "chunk_id": cid,
                    "source_id": meta.get("source_id", cid),
                    "source_name": meta.get("source_name", "Retrieved Source"),
                    "tier": int(meta.get("tier", c.get("tier", 1))),
                    "year": int(meta.get("year", c.get("year", 1991))),
                    "raw_text": c.get("raw_text", ""),
                    "snyder_text": c.get("raw_text", ""),
                }

        # 2. LLM Synthesis with Truncation Bypass
        llm_response, llm_meta = self.llm_client.generate_response(
            question=question,
            retrieved_chunks=retrieved_chunks,
            mock_response_generator=mock_response_generator,
        )

        # 3. Mathematical Confidence Evaluation
        confidence_report = self.confidence_engine.evaluate(
            llm_response=llm_response,
            chunk_database=self.chunk_database,
        )

        result = {
            "question": question,
            "narrative": llm_response.narrative,
            "timeline_shifts": [t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in llm_response.timeline_shifts],
            "citations": [c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in llm_response.citations],
            "flags": llm_response.flags.model_dump() if hasattr(llm_response.flags, "model_dump") else llm_response.flags.dict(),
            "confidence": confidence_report.model_dump() if hasattr(confidence_report, "model_dump") else confidence_report.dict(),
            "retrieval_metadata": {
                "chunks_retrieved": len(retrieved_chunks),
                "years_covered": [c.get("year", 1991) for c in retrieved_chunks],
                "llm_attempts": llm_meta.get("attempts", 1),
                "truncation_bypasses": llm_meta.get("truncation_bypasses", 0),
            }
        }

        return result
