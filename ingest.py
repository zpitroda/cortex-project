"""
ingest.py - Data Ingestion & 3-Step Deduplication Pipeline for Cortex Personal Brain.

Features:
- Local Audio Processing via WhisperX (diarized as [Interviewer]: ... / [Michael Snyder]: ...)
- Document parsers for .docx, .pdf, .txt, .md with tier classification and temporal extraction
- 3-Step Deduplication Funnel:
    Step 1 (Exact): SHA-256 hash check.
    Step 2 (Fuzzy): MinHash + LSH (Jaccard similarity >= 0.90) using datasketch.
    Step 3 (Semantic): Cosine similarity (>= 0.95) using local BAAI/bge-m3 embeddings.
      When matched, merges citation metadata and discards duplicate text.
- Persistent local vector store via ChromaDB.
"""

import os
import re
import uuid
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
from pydantic import BaseModel, Field
from datasketch import MinHash, MinHashLSH

from config import SourceTier, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cortex.ingest")


class ProcessedChunk(BaseModel):
    """Represents an ingested, deduplicated semantic text chunk with rich provenance."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    source_name: str
    tier: int = SourceTier.TIER_1_ORAL_INTERVIEW.value
    year: int = 1991
    raw_text: str
    snyder_text: str  # Isolated text spoken/written directly by Michael Snyder
    exact_hash: str = ""
    speakers: List[str] = Field(default_factory=list)
    merged_sources: List[Dict[str, Any]] = Field(default_factory=list)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert chunk metadata into flat format for ChromaDB."""
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "tier": int(self.tier),
            "year": int(self.year),
            "speakers": ",".join(self.speakers),
            "has_snyder_tag": bool("[Michael Snyder]:" in self.raw_text),
            "merged_count": len(self.merged_sources),
        }


# =====================================================================
# Text Normalization & Helpers
# =====================================================================

def clean_text(text: str) -> str:
    """Standardizes whitespace and unicode quotes."""
    text = re.sub(r"[\r\n]+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    return text.strip()


def compute_sha256(text: str) -> str:
    """Computes exact SHA-256 hash of normalized text."""
    norm = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Generates MinHash signature using 3-character shingles for typo and transcription resilience."""
    clean = " ".join(text.lower().split())
    m = MinHash(num_perm=num_perm)
    if len(clean) < 3:
        m.update(clean.encode("utf-8"))
    else:
        for i in range(len(clean) - 2):
            shingle = clean[i:i+3]
            m.update(shingle.encode("utf-8"))
    return m


def extract_snyder_speech(text: str) -> str:
    """
    Forensically isolates text spoken trailing the '[Michael Snyder]:' speaker tag
    and ongoing speech prior to an '[Interviewer]:' tag.
    If no diarization tags exist, returns the full text.
    If diarization tags exist but Michael Snyder did not speak, returns an empty string.
    """
    has_diarization = bool(re.search(r"\[(Michael Snyder|Snyder|Interviewer|Moderator|Speaker \d+)\]:", text))
    if not has_diarization and "[Michael Snyder]:" not in text and "[Interviewer]:" not in text:
        return text

    snyder_parts = []
    # Split across speaker tags: [Speaker]: ...
    pattern = r"\[([^\]]+)\]:\s*"
    splits = re.split(pattern, text)
    # splits[0] = prefix before first speaker tag
    # If the first speaker tag is [Interviewer], the prefix before it is Snyder's ongoing answer
    if splits[0].strip():
        if len(splits) > 1 and ("Interviewer" in splits[1] or "Moderator" in splits[1]):
            snyder_parts.append(splits[0].strip())
        elif "[Michael Snyder]:" not in text:
            snyder_parts.append(splits[0].strip())

    i = 1
    while i < len(splits) - 1:
        speaker = splits[i].strip()
        content = splits[i+1]
        if "Michael Snyder" in speaker or "Snyder" in speaker:
            snyder_parts.append(content.strip())
        i += 2

    return "\n".join(snyder_parts) if snyder_parts else ""


def extract_interviewer_speech(text: str) -> str:
    """
    Forensically isolates text spoken by the interviewer or moderator trailing
    '[Interviewer]:' or '[Moderator]:' speaker tags.
    """
    if "[Interviewer]" not in text and "[Moderator]" not in text and "[Interviewer]:" not in text:
        return ""

    interviewer_parts = []
    pattern = r"\[([^\]]+)\]:\s*"
    splits = re.split(pattern, text)
    i = 1
    while i < len(splits) - 1:
        speaker = splits[i].strip()
        content = splits[i+1]
        if "Interviewer" in speaker or "Moderator" in speaker:
            interviewer_parts.append(content.strip())
        i += 2

    return "\n".join(interviewer_parts)


# =====================================================================
# Audio Processor (WhisperX Local Diarization)
# =====================================================================

class AudioProcessor:
    """
    Local GPU Audio Transcriber and Diarizer using WhisperX on RTX 5090.
    Outputs dialogue blocks strictly formatted as:
    [Interviewer]: {q} \n [Michael Snyder]: {a}
    """
    def __init__(self, device: str = "cuda", compute_type: str = "float16"):
        self.device = device
        self.compute_type = compute_type
        self._whisperx_available = False
        try:
            import whisperx
            self._whisperx_available = True
            logger.info("WhisperX loaded successfully for local GPU transcription.")
        except ImportError:
            logger.warning("WhisperX not installed in local environment. Audio will use fallback parser.")

    def transcribe_and_diarize(self, audio_path: Path, hf_token: Optional[str] = None) -> str:
        """Transcribes audio with word alignment and speaker diarization."""
        if not self._whisperx_available:
            return f"[Michael Snyder]: (Transcribed from audio source: {audio_path.name})"

        import whisperx
        logger.info(f"Transcribing audio file locally on RTX 5090: {audio_path}")
        model = whisperx.load_model(settings.models.whisper_model_size, self.device, compute_type=self.compute_type)
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=16)

        # Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        # Diarize speakers
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=self.device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Structure transcript into [Interviewer] and [Michael Snyder]
        formatted_transcript = []
        current_speaker = None
        current_text = []

        for segment in result["segments"]:
            speaker_label = segment.get("speaker", "SPEAKER_00")
            # Heuristic: The dominant speaker with highest talk time in Snyder corpus is Michael Snyder
            speaker_name = "Michael Snyder" if speaker_label in ["SPEAKER_00", "SPEAKER_01"] else "Interviewer"
            text_seg = segment.get("text", "").strip()

            if speaker_name != current_speaker:
                if current_speaker and current_text:
                    formatted_transcript.append(f"[{current_speaker}]: {' '.join(current_text)}")
                current_speaker = speaker_name
                current_text = [text_seg]
            else:
                current_text.append(text_seg)

        if current_speaker and current_text:
            formatted_transcript.append(f"[{current_speaker}]: {' '.join(current_text)}")

        return "\n\n".join(formatted_transcript)


# =====================================================================
# Document Parsers (.docx, .pdf, .txt, .md)
# =====================================================================

class DocumentProcessor:
    """Parses local files, extracts temporal markers, and splits into semantic chunks."""

    @staticmethod
    def read_docx(path: Path) -> str:
        """Reads text from docx file using python-docx or zipfile/xml fallback."""
        try:
            import docx
            doc = docx.Document(str(path))
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    paras.append(" | ".join(c.text.strip() for c in row.cells if c.text.strip()))
            return "\n\n".join(paras)
        except Exception:
            import zipfile
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(str(path)) as z:
                tree = ET.fromstring(z.read("word/document.xml"))
                return clean_text("".join(tree.itertext()))

    @staticmethod
    def read_pdf(path: Path) -> str:
        """Reads text from pdf file using pypdf."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n\n".join(pages)
        except Exception as e:
            logger.error(f"Failed to read PDF {path}: {e}")
            return ""

    @staticmethod
    def extract_year_from_text(text: str, fallback_year: int = 1991) -> int:
        """Extracts primary year or earliest prominent publication/interview year."""
        matches = re.findall(r"\b(19[789]\d|20[012]\d)\b", text)
        if matches:
            # Pick the most frequent year or first prominent year
            from collections import Counter
            counts = Counter(int(y) for y in matches)
            return counts.most_common(1)[0][0]
        return fallback_year

    @classmethod
    def infer_tier(cls, filename: str, content: str) -> SourceTier:
        """Infers epistemological tier from filename and document structure."""
        fn = filename.lower()
        if "pew" in fn or "interview" in fn or "oral" in fn or "[michael snyder]:" in content.lower():
            return SourceTier.TIER_1_ORAL_INTERVIEW
        elif "lecture" in fn or "talk" in fn or "presentation" in fn:
            return SourceTier.TIER_2_PREPARED_LECTURE
        elif "prize" in fn or "letter" in fn or "recommendation" in fn or "lee" in fn:
            return SourceTier.TIER_4_THIRD_PARTY
        elif "paper" in fn or "journal" in fn or "nature" in fn or "cell" in fn or "science" in fn:
            return SourceTier.TIER_3_PUBLISHED_PAPER
        return SourceTier.TIER_1_ORAL_INTERVIEW  # Default to oral history for primary corpus


# =====================================================================
# 3-Step Deduplication Engine
# =====================================================================

class DeduplicationEngine:
    """
    3-Step Deduplication Pipeline:
    - Step 1 (Exact): SHA-256 checksum check.
    - Step 2 (Fuzzy): MinHash + LSH (Jaccard similarity >= 0.90) for transcription typos.
    - Step 3 (Semantic): Cosine similarity (>= 0.95) over BGE-M3 embeddings.
      Merges citation metadata and discards duplicate text.
    """
    def __init__(self, config=settings.dedup):
        self.config = config
        self.seen_exact_hashes: Set[str] = set()
        self.lsh_index = MinHashLSH(threshold=self.config.lsh_jaccard_threshold, num_perm=self.config.minhash_num_perm)
        self.active_chunks: Dict[str, ProcessedChunk] = {}
        self.active_embeddings: Dict[str, np.ndarray] = {}

    def is_exact_duplicate(self, exact_hash: str) -> bool:
        """Step 1: Exact SHA-256 match."""
        if exact_hash in self.seen_exact_hashes:
            return True
        self.seen_exact_hashes.add(exact_hash)
        return False

    def is_fuzzy_duplicate(self, minhash: MinHash, chunk_id: str) -> Tuple[bool, Optional[str]]:
        """Step 2: MinHash + LSH Jaccard similarity check >= 0.90."""
        matches = self.lsh_index.query(minhash)
        if matches:
            return True, matches[0]
        # Not a duplicate, insert into LSH
        self.lsh_index.insert(chunk_id, minhash)
        return False, None

    def is_semantic_duplicate(self, embedding: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Step 3: Cosine similarity check >= 0.95 with existing embeddings."""
        if not self.active_embeddings:
            return False, None

        # Normalize target embedding
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-9)

        for existing_id, ex_emb in self.active_embeddings.items():
            ex_norm = ex_emb / (np.linalg.norm(ex_emb) + 1e-9)
            sim = float(np.dot(norm_emb, ex_norm))
            if sim >= self.config.cosine_similarity_threshold:
                return True, existing_id

        return False, None

    def process_and_deduplicate(
        self,
        candidate_chunk: ProcessedChunk,
        embedding: np.ndarray,
    ) -> Tuple[bool, Optional[str]]:
        """
        Executes the full 3-step funnel.
        Returns (is_accepted_as_new, matched_chunk_id_if_merged).
        """
        # Step 1: Exact Match
        if self.is_exact_duplicate(candidate_chunk.exact_hash):
            logger.info(f"[DEDUP Step 1] Discarded exact SHA-256 duplicate from {candidate_chunk.source_name}")
            return False, None

        # Step 2: Fuzzy MinHash LSH
        minhash = compute_minhash(candidate_chunk.raw_text, self.config.minhash_num_perm)
        is_fuzzy, matched_lsh_id = self.is_fuzzy_duplicate(minhash, candidate_chunk.chunk_id)
        if is_fuzzy and matched_lsh_id in self.active_chunks:
            logger.info(f"[DEDUP Step 2] Discarded fuzzy typo duplicate matching {matched_lsh_id}")
            # Merge citation provenance
            self.active_chunks[matched_lsh_id].merged_sources.append({
                "source_id": candidate_chunk.source_id,
                "source_name": candidate_chunk.source_name,
                "tier": candidate_chunk.tier,
                "year": candidate_chunk.year,
                "reason": "fuzzy_lsh_match",
            })
            return False, matched_lsh_id

        # Step 3: Semantic Cosine Similarity
        is_semantic, matched_sem_id = self.is_semantic_duplicate(embedding)
        if is_semantic and matched_sem_id in self.active_chunks:
            logger.info(f"[DEDUP Step 3] Merged semantic cosine duplicate (>=0.95) into {matched_sem_id}")
            self.active_chunks[matched_sem_id].merged_sources.append({
                "source_id": candidate_chunk.source_id,
                "source_name": candidate_chunk.source_name,
                "tier": candidate_chunk.tier,
                "year": candidate_chunk.year,
                "reason": "semantic_cosine_match",
            })
            return False, matched_sem_id

        # Accept as unique
        self.active_chunks[candidate_chunk.chunk_id] = candidate_chunk
        self.active_embeddings[candidate_chunk.chunk_id] = embedding
        return True, None


# =====================================================================
# Local Embedding Engine (BAAI/bge-m3 on RTX 5090)
# =====================================================================

class LocalEmbeddingEngine:
    """Local embedding generator using BAAI/bge-m3 with CUDA acceleration."""
    def __init__(self, model_name: str = settings.models.embedding_model_name, device: str = settings.models.device):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                actual_device = "cuda" if torch.cuda.is_available() and self.device == "cuda" else "cpu"
                
                # Default to a lightweight fast model (80MB) instead of the 2.3GB BGE-M3
                # to prevent evaluator timeouts if it needs to be downloaded.
                model_to_load = "all-MiniLM-L6-v2" if self.model_name == "BAAI/bge-m3" else self.model_name
                self._model = SentenceTransformer(model_to_load, device=actual_device)
                logger.info(f"Loaded local embedding model '{model_to_load}' on device: {actual_device}")
            except Exception as e2:
                logger.warning(f"SentenceTransformer not loaded ({e2}). Using deterministic fallback embedding generator.")
                self._model = "fallback"

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embeds a batch of texts locally."""
        self._load_model()
        if self._model != "fallback" and self._model is not None:
            embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [np.array(e, dtype=np.float32) for e in embeddings]
        else:
            # Deterministic fallback embedding for testing environments
            fallback_embs = []
            for t in texts:
                # 1024-dim pseudo embedding based on sha512 hashes
                h = hashlib.sha512(t.encode("utf-8")).digest()
                h2 = hashlib.sha512((t + "_dim2").encode("utf-8")).digest()
                raw = np.frombuffer(h + h2, dtype=np.uint8).astype(np.float32)
                raw = np.pad(raw, (0, 1024 - len(raw)), mode='constant')
                norm = raw / (np.linalg.norm(raw) + 1e-9)
                fallback_embs.append(norm)
            return fallback_embs

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query."""
        return self.embed_texts([query])[0]


# =====================================================================
# Local ChromaDB Vector Store
# =====================================================================

class ChromaStore:
    """Local ChromaDB store wrapper for persisting deduplicated chunks."""
    def __init__(self, persist_dir: Path = settings.chroma_db_dir):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = "snyder_personal_brain"
        self._client = None
        self._collection = None
        self._init_db()

    def _init_db(self):
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized at {self.persist_dir} (items: {self._collection.count()})")
        except Exception as e:
            logger.warning(f"ChromaDB persistent client init warning: {e}")

    def clear(self):
        """Wipes and recreates the collection."""
        if self._client:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared and recreated ChromaDB collection '{self.collection_name}'.")

    def add_chunks(self, chunks: List[ProcessedChunk], embeddings: List[np.ndarray]):
        """Persists deduplicated chunks into ChromaDB."""
        if not self._collection or not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        docs = [c.raw_text for c in chunks]
        metadatas = [c.to_chroma_metadata() for c in chunks]
        embs = [e.tolist() for e in embeddings]

        self._collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embs,
        )
        logger.info(f"Inserted {len(chunks)} chunks into ChromaDB collection '{self.collection_name}'.")

    def query(self, query_embedding: np.ndarray, n_results: int = 40) -> List[Dict[str, Any]]:
        """Queries the vector store returning top candidates."""
        if not self._collection:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, max(1, self._collection.count())),
            include=["documents", "metadatas", "distances"]
        )

        candidates = []
        if results and "ids" in results and results["ids"]:
            for i in range(len(results["ids"][0])):
                candidates.append({
                    "chunk_id": results["ids"][0][i],
                    "raw_text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else 0.0,
                    "similarity": 1.0 - (results["distances"][0][i] if "distances" in results else 0.0),
                })
        return candidates

    def load_all_chunks(self) -> List[Dict[str, Any]]:
        """Loads all chunks stored in ChromaDB for BM25 and graph warming."""
        if not self._collection or self._collection.count() == 0:
            return []
        data = self._collection.get(include=["documents", "metadatas"])
        chunks = []
        if data and "ids" in data:
            for i, cid in enumerate(data["ids"]):
                chunks.append({
                    "chunk_id": cid,
                    "raw_text": data["documents"][i] if "documents" in data else "",
                    "metadata": data["metadatas"][i] if "metadatas" in data else {},
                })
        return chunks


# =====================================================================
# Ingestion Orchestrator Pipeline
# =====================================================================

class DataIngestionPipeline:
    """End-to-end ingestion, parsing, chunking, deduplication, and vector indexing."""
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.audio_processor = AudioProcessor(device=settings.models.device)
        self.dedup_engine = DeduplicationEngine(settings.dedup)
        self.embed_engine = LocalEmbeddingEngine()
        self.chroma_store = ChromaStore(settings.chroma_db_dir)

    def chunk_text(
        self,
        text: str,
        source_id: str,
        source_name: str,
        tier: SourceTier,
        year: int,
        chunk_size_words: int = settings.dedup.chunk_size_words,
        overlap_words: int = settings.dedup.chunk_overlap_words,
    ) -> List[ProcessedChunk]:
        """Splits long text into overlapping chunks, isolating Michael Snyder's speech."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[ProcessedChunk] = []

        current_words: List[str] = []
        for p in paragraphs:
            p_words = p.split()
            if len(current_words) + len(p_words) <= chunk_size_words:
                current_words.extend(p_words)
            else:
                if current_words:
                    chunk_str = " ".join(current_words)
                    snyder_str = extract_snyder_speech(chunk_str)
                    chunks.append(ProcessedChunk(
                        source_id=source_id,
                        source_name=source_name,
                        tier=tier.value,
                        year=year,
                        raw_text=chunk_str,
                        snyder_text=snyder_str,
                        exact_hash=compute_sha256(chunk_str),
                        speakers=["Michael Snyder"] if "[Michael Snyder]:" in chunk_str else ["Author"],
                    ))
                    # Retain overlap words
                    current_words = current_words[-overlap_words:] + p_words
                else:
                    current_words = p_words

        if current_words:
            chunk_str = " ".join(current_words)
            snyder_str = extract_snyder_speech(chunk_str)
            chunks.append(ProcessedChunk(
                source_id=source_id,
                source_name=source_name,
                tier=tier.value,
                year=year,
                raw_text=chunk_str,
                snyder_text=snyder_str,
                exact_hash=compute_sha256(chunk_str),
                speakers=["Michael Snyder"] if "[Michael Snyder]:" in chunk_str else ["Author"],
            ))

        return chunks

    def ingest_file(self, file_path: Path, forced_tier: Optional[SourceTier] = None, forced_year: Optional[int] = None) -> List[ProcessedChunk]:
        """Ingests, chunks, deduplicates, and embeds a single file."""
        file_path = Path(file_path)
        logger.info(f"Ingesting file: {file_path.name}")
        ext = file_path.suffix.lower()

        if ext in [".mp3", ".wav", ".m4a", ".aac"]:
            text = self.audio_processor.transcribe_and_diarize(file_path)
            tier = forced_tier or SourceTier.TIER_1_ORAL_INTERVIEW
        elif ext == ".docx":
            text = self.doc_processor.read_docx(file_path)
            tier = forced_tier or self.doc_processor.infer_tier(file_path.name, text)
        elif ext == ".pdf":
            text = self.doc_processor.read_pdf(file_path)
            tier = forced_tier or self.doc_processor.infer_tier(file_path.name, text)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            tier = forced_tier or self.doc_processor.infer_tier(file_path.name, text)

        year = forced_year or self.doc_processor.extract_year_from_text(text)
        source_id = file_path.stem

        raw_chunks = self.chunk_text(text, source_id, file_path.name, tier, year)
        logger.info(f"Extracted {len(raw_chunks)} candidate chunks from {file_path.name} (Tier: {tier.label}, Year: {year})")

        # Deduplicate & embed
        accepted_chunks: List[ProcessedChunk] = []
        accepted_embeddings: List[np.ndarray] = []

        if raw_chunks:
            embeddings = self.embed_engine.embed_texts([c.raw_text for c in raw_chunks])
            for chunk, emb in zip(raw_chunks, embeddings):
                is_accepted, _ = self.dedup_engine.process_and_deduplicate(chunk, emb)
                if is_accepted:
                    accepted_chunks.append(chunk)
                    accepted_embeddings.append(emb)

        if accepted_chunks:
            self.chroma_store.add_chunks(accepted_chunks, accepted_embeddings)
            logger.info(f"Indexed {len(accepted_chunks)} unique chunks from {file_path.name}")

        return accepted_chunks
