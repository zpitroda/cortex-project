"""
config.py - Central Configuration & Schema Definitions for Cortex Personal Brain.

Handles settings, source tier rankings, thresholds for 3-step deduplication,
local model configurations (BGE-M3, WhisperX, spaCy), custom API endpoint headers,
and mathematical confidence parameters.
"""

import os
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass



class SourceTier(IntEnum):
    """
    Source reliability tier hierarchy for Michael Snyder's personal brain.
    - Tier 1: Raw Audio / Diarized Oral Histories (highest epistemological weight: Base +40)
    - Tier 2: Prepared Lectures / Recorded Talks (Base +30)
    - Tier 3: Published Peer-Reviewed Papers (tidy post-hoc descriptions: Base +15)
    - Tier 4: Third-Party Accounts / Recommendation Letters / Press (Base +5)
    """
    TIER_1_ORAL_INTERVIEW = 1
    TIER_2_PREPARED_LECTURE = 2
    TIER_3_PUBLISHED_PAPER = 3
    TIER_4_THIRD_PARTY = 4

    @property
    def base_score(self) -> float:
        mapping = {
            SourceTier.TIER_1_ORAL_INTERVIEW: 35.0,
            SourceTier.TIER_2_PREPARED_LECTURE: 25.0,
            SourceTier.TIER_3_PUBLISHED_PAPER: 10.0,
            SourceTier.TIER_4_THIRD_PARTY: 0.0,
        }
        return mapping[self]

    @property
    def label(self) -> str:
        mapping = {
            SourceTier.TIER_1_ORAL_INTERVIEW: "Tier 1: Oral History / Unrehearsed Audio",
            SourceTier.TIER_2_PREPARED_LECTURE: "Tier 2: Prepared Lecture / Talk",
            SourceTier.TIER_3_PUBLISHED_PAPER: "Tier 3: Published Paper",
            SourceTier.TIER_4_THIRD_PARTY: "Tier 4: Third-Party Account / Colleague Letter",
        }
        return mapping[self]


# =====================================================================
# Pydantic Schemas for Strict Structured LLM Output
# =====================================================================

class TimelineShift(BaseModel):
    year: Any = Field(1991, description="The specific year or time period of the belief/approach.")
    belief: str = Field(..., description="Summary of what Snyder believed, attempted, or decided.")
    chunk_id: str = Field(default="", description="The provenance chunk ID grounding this timeline shift.")

    def get_year_int(self) -> int:
        try:
            import re
            m = re.search(r"\d{4}", str(self.year))
            return int(m.group(0)) if m else 1991
        except Exception:
            return 1991


class Citation(BaseModel):
    chunk_id: str = Field(default="", description="The source chunk ID providing direct evidence.")
    exact_quote: str = Field(..., description="Exact, verbatim substring extracted from Michael Snyder's speech/writing.")


class OutputFlags(BaseModel):
    is_inferred: bool = Field(False, description="True if the answer relies on extrapolation or speculation beyond direct quotes.")
    thin_record: bool = Field(False, description="True if documentation on this period or topic is sparse or single-source.")


class LLMResponseSchema(BaseModel):
    """Strict JSON output schema enforced on Claude 4.5 Sonnet."""
    narrative: str = Field(..., description="Detailed chronological explanation of his scientific reasoning, evolution, and pivots.")
    timeline_shifts: List[TimelineShift] = Field(default_factory=list, description="Key chronological milestones and beliefs.")
    citations: List[Citation] = Field(default_factory=list, description="Verbatim quote citations linked to source chunk IDs.")
    flags: OutputFlags = Field(default_factory=OutputFlags, description="Epistemological flags for inference and record density.")
    EOF_KEY: bool = Field(True, alias="__EOF__", description="Deterministic completion token to detect silent truncation.")

    class Config:
        populate_by_name = True


# =====================================================================
# System Configurations
# =====================================================================

class DeduplicationConfig(BaseModel):
    """Parameters for the 3-step exact -> fuzzy -> semantic deduplication funnel."""
    minhash_num_perm: int = 128
    lsh_jaccard_threshold: float = 0.90       # Step 2: LSH Jaccard similarity threshold
    cosine_similarity_threshold: float = 0.95 # Step 3: BGE-M3 Cosine similarity threshold
    chunk_size_words: int = 250
    chunk_overlap_words: int = 40


class ModelConfig(BaseModel):
    """Local compute models for RTX 5090 acceleration."""
    model_config = {"protected_namespaces": ()}
    embedding_model_name: str = "BAAI/bge-m3"
    spacy_model_name: str = "en_core_web_sm"
    device: str = "cuda"
    whisper_model_size: str = "base"
    whisper_compute_type: str = "float16"


class BM25Config(BaseModel):
    """Parameters for sparse BM25 lexical retrieval."""
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60


class RerankerConfig(BaseModel):
    """Parameters for local Cross-Encoder re-ranking on RTX 5090."""
    model_config = {"protected_namespaces": ()}
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n_rerank: int = 30


class RetrievalConfig(BaseModel):
    """Anti-Flattening GraphRAG parameters."""
    candidate_pool_size: int = 40
    max_llm_chunks: int = 15  # Strict constraint to honor the 300,000 token ceiling
    temporal_span_bins: int = 5
    bm25: BM25Config = BM25Config()
    reranker: RerankerConfig = RerankerConfig()


class LLMConfig(BaseModel):
    """Custom Claude 4.5 Sonnet and Qwen 3.8 endpoint configuration."""
    model_config = {"protected_namespaces": ()}
    api_url: str = Field(default_factory=lambda: os.getenv("CORTEX_API_URL", "https://api.cortex.bio/v1/chat/completions"))
    api_key: str = Field(default_factory=lambda: os.getenv("CORTEX_API_KEY", ""))
    mode: str = Field(default_factory=lambda: os.getenv("CORTEX_SESSION_MODE", os.getenv("CORTEX_MODE", "")))  # Empty string defaults to scored session (199k budget)
    max_retries: int = 5
    model_name: str = Field(default_factory=lambda: os.getenv("CORTEX_MODEL_NAME", "qwen3.8"))
    thinking_level: str = Field(default_factory=lambda: os.getenv("CORTEX_THINKING_LEVEL", "medium"))
    reasoning_effort: str = Field(default_factory=lambda: os.getenv("CORTEX_REASONING_EFFORT", "medium"))
    temperature: float = 0.0
    timeout_seconds: int = 60


class ConfidenceConfig(BaseModel):
    """Mathematical Confidence Engine scoring matrix."""
    tier_1_base: float = 35.0
    tier_2_base: float = 25.0
    tier_3_base: float = 10.0
    tier_4_base: float = 0.0

    multi_year_corroboration: float = 30.0
    same_year_multi_source: float = 20.0
    single_source: float = 10.0

    exact_quote_match: float = 25.0
    partial_quote_match: float = 12.5
    unverified_quote: float = 0.0

    recency_post_2010: float = 10.0
    recency_2000_2009: float = 5.0
    recency_pre_2000: float = 0.0

    inference_penalty: float = 50.0
    thin_record_penalty: float = 40.0


class CortexSettings(BaseModel):
    """Global configuration aggregator."""
    base_dir: Path = Path("l:/cortex")
    chroma_db_dir: Path = Path("l:/cortex/data/chromadb")
    graph_storage_path: Path = Path("l:/cortex/data/processed/temporal_graph.json")
    raw_data_dir: Path = Path("l:/cortex/data/raw")
    processed_data_dir: Path = Path("l:/cortex/data/processed")

    dedup: DeduplicationConfig = DeduplicationConfig()
    models: ModelConfig = ModelConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()


settings = CortexSettings()
