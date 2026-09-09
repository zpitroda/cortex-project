"""
retrieval.py - Anti-Flattening GraphRAG & Chronological Retrieval Funnel.

Standard RAG flattens 35 years of scientific thought into a single present tense.
This module extracts an Entity-Temporal graph locally using spaCy/Regex into NetworkX,
queries the graph for concept temporal envelopes, searches ChromaDB for semantic matches,
and applies stratified chronological sampling to ensure Claude 4.5 Sonnet sees the full
historical evolution of Michael Snyder's thinking (capped at max 15 chunks for the 300k budget).
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

import networkx as nx
import numpy as np

from config import SourceTier, settings
from ingest import ProcessedChunk, ChromaStore, LocalEmbeddingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cortex.retrieval")


# =====================================================================
# Local Entity & Temporal Graph Extractor
# =====================================================================

class TemporalGraphExtractor:
    """
    Extracts biological entities, technologies, mentors, and temporal anchors
    locally into a NetworkX directed knowledge graph.
    """
    # Key historical entities and topics across Snyder's 35-year career
    KNOWN_ENTITIES = {
        # Organisms
        "yeast": "Organism",
        "saccharomyces cerevisiae": "Organism",
        "drosophila": "Organism",
        "fruit fly": "Organism",
        "human": "Organism",
        "mouse": "Organism",
        "e. coli": "Organism",

        # Technologies & Innovations
        "transposon": "Technology",
        "transposon mutagenesis": "Technology",
        "epitope tagging": "Technology",
        "microarray": "Technology",
        "dna microarray": "Technology",
        "chip-chip": "Technology",
        "chip-seq": "Technology",
        "rna-seq": "Technology",
        "mass spectrometry": "Technology",
        "ipop": "Technology",
        "personal omics": "Technology",
        "wearables": "Technology",
        "continuous glucose monitor": "Technology",
        "cgm": "Technology",

        # Mentors, Colleagues & Institutions
        "ron davis": "Person",
        "david botstein": "Person",
        "paul berg": "Person",
        "charles lee": "Person",
        "stanford": "Institution",
        "yale": "Institution",
        "caltech": "Institution",
        "pew": "Organization",

        # Philosophical / Career Themes
        "competition": "Theme",
        "sharing reagents": "Theme",
        "large science": "Theme",
        "small science": "Theme",
        "tenure": "Theme",
        "functional genomics": "Theme",
        "systems biology": "Theme",
    }

    def __init__(self, graph_path: Path = settings.graph_storage_path):
        self.graph_path = graph_path
        self.graph = nx.DiGraph()
        self._load_nlp()
        self.load_graph()

    def _load_nlp(self):
        try:
            import spacy
            self.nlp = spacy.load(settings.models.spacy_model_name)
        except Exception:
            self.nlp = None

    def load_graph(self):
        """Loads serialized graph if present."""
        if self.graph_path.exists():
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                logger.info(f"Loaded temporal graph from {self.graph_path} ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")
            except Exception as e:
                logger.warning(f"Could not load existing graph ({e}). Initializing fresh graph.")
                self.graph = nx.DiGraph()

    def save_graph(self):
        """Persists graph to disk."""
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved temporal graph to {self.graph_path}")

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extracts known domain entities and general NER named entities."""
        entities: Set[Tuple[str, str]] = set()
        lower_text = text.lower()

        # Domain dictionary matching with word boundaries
        for entity_name, entity_type in self.KNOWN_ENTITIES.items():
            if re.search(r"\b" + re.escape(entity_name) + r"\b", lower_text):
                entities.add((entity_name.title(), entity_type))

        # spaCy NER fallback
        if self.nlp:
            doc = self.nlp(text[:2000])
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                    entities.add((ent.text.strip(), ent.label_))

        return list(entities)

    def extract_years(self, text: str, default_year: int) -> List[int]:
        """Extracts explicitly mentioned years in text."""
        years = set(int(y) for y in re.findall(r"\b(19[789]\d|20[012]\d)\b", text))
        if default_year:
            years.add(default_year)
        return sorted(list(years))

    # Epistemic paradigm shift patterns
    SHIFT_PATTERNS = [
        (r"(?:moved|switched|transitioned|pivoted|shifted)\s+(?:away\s+)?from\s+([a-zA-Z\s\-]+?)\s+to\s+([a-zA-Z\s\-]+)", "SUPERSEDED_BY"),
        (r"instead\s+of\s+([a-zA-Z\s\-]+?)[,\s]+(?:we|I)\s+(?:worked on|chose|focused on|used)\s+([a-zA-Z\s\-]+)", "SUPERSEDED_BY"),
        (r"(?:used to think|thought early on)\s+([a-zA-Z\s\-]+?)\s+but\s+(?:later realized|found)\s+([a-zA-Z\s\-]+)", "CHANGED_MIND_ON"),
        (r"([a-zA-Z\s\-]+?)\s+(?:evolved into|led to|paved the way for)\s+([a-zA-Z\s\-]+)", "EVOLVED_INTO"),
        (r"([a-zA-Z\s\-]+?)\s+were\s+too\s+slow[,\s]+(?:so\s+)?(?:I|we)\s+(?:moved to|switched to|chose)\s+([a-zA-Z\s\-]+)", "SUPERSEDED_BY"),
    ]

    def detect_epistemic_transitions(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Detects epistemic pivots where Michael Snyder changed his mind or methodology.
        Returns list of (source_concept, target_concept, relation) tuples.
        """
        transitions: List[Tuple[str, str, str]] = []
        lower_text = text.lower()

        # Regex heuristic extraction
        for pattern, rel in self.SHIFT_PATTERNS:
            for match in re.finditer(pattern, lower_text):
                src = match.group(1).strip()
                tgt = match.group(2).strip()

                # Clean up punctuation and stop words
                src_clean = re.sub(r"^(the|a|an|working on|studying)\s+", "", src).strip()
                tgt_clean = re.sub(r"^(the|a|an|working on|studying)\s+", "", tgt).strip()

                # Match against known entity ontology if possible
                matched_src = None
                matched_tgt = None
                for k in self.KNOWN_ENTITIES:
                    if k in src_clean:
                        matched_src = k.title()
                    if k in tgt_clean:
                        matched_tgt = k.title()

                final_src = matched_src or src_clean[:30].title()
                final_tgt = matched_tgt or tgt_clean[:30].title()

                if final_src and final_tgt and final_src != final_tgt:
                    transitions.append((final_src, final_tgt, rel))

        # Explicit historical transitions
        if ("drosophila" in lower_text or "flies" in lower_text) and "yeast" in lower_text:
            if any(w in lower_text for w in ["slow", "switch", "moved", "transition", "instead"]):
                transitions.append(("Drosophila", "Yeast", "SUPERSEDED_BY"))

        if "microarray" in lower_text and ("rna-seq" in lower_text or "sequencing" in lower_text):
            if any(w in lower_text for w in ["switch", "replaced", "evolved", "moved to", "better"]):
                transitions.append(("Microarray", "Rna-Seq", "EVOLVED_INTO"))

        if ("genetics" in lower_text or "genome" in lower_text) and ("wearable" in lower_text or "continuous glucose" in lower_text or "cgm" in lower_text):
            if any(w in lower_text for w in ["static", "dynamic", "shift", "future", "continuous", "monitoring"]):
                transitions.append(("Static Genetics", "Continuous Wearables", "EVOLVED_INTO"))

        return transitions

    def index_chunk(self, chunk: ProcessedChunk):
        """Builds graph nodes and edges for a chunk, including epistemic shift transitions."""
        chunk_node = f"chunk:{chunk.chunk_id}"
        self.graph.add_node(
            chunk_node,
            node_type="chunk",
            chunk_id=chunk.chunk_id,
            source_name=chunk.source_name,
            tier=chunk.tier,
            year=chunk.year,
        )

        # Year nodes
        mentioned_years = self.extract_years(chunk.raw_text, chunk.year)
        for y in mentioned_years:
            year_node = f"year:{y}"
            self.graph.add_node(year_node, node_type="year", value=y)
            self.graph.add_edge(chunk_node, year_node, relation="anchored_in_year")

        # Entity nodes
        entities = self.extract_entities(chunk.raw_text)
        for ent_name, ent_type in entities:
            ent_node = f"entity:{ent_name.lower()}"
            self.graph.add_node(ent_node, node_type="entity", name=ent_name, category=ent_type)
            self.graph.add_edge(chunk_node, ent_node, relation="mentions")

            for y in mentioned_years:
                year_node = f"year:{y}"
                edge_data = self.graph.get_edge_data(ent_node, year_node)
                current_count = edge_data.get("count", 0) if edge_data else 0
                self.graph.add_edge(ent_node, year_node, count=current_count + 1)

        # Epistemic Transitions
        transitions = self.detect_epistemic_transitions(chunk.raw_text)
        for src_c, tgt_c, rel in transitions:
            src_node = f"entity:{src_c.lower()}"
            tgt_node = f"entity:{tgt_c.lower()}"
            self.graph.add_node(src_node, node_type="entity", name=src_c, category="Concept")
            self.graph.add_node(tgt_node, node_type="entity", name=tgt_c, category="Concept")

            self.graph.add_edge(
                src_node,
                tgt_node,
                relation=rel,
                year=chunk.year,
                chunk_id=chunk.chunk_id,
                evidence=chunk.raw_text[:120],
            )
            # Link chunk to the shift edge
            self.graph.add_edge(chunk_node, tgt_node, relation="documents_shift_to")
            logger.info(f"[GRAPH TRANSITION] Indexed epistemic shift: ({src_c}) --[{rel}]--> ({tgt_c}) in Year {chunk.year}")

    def get_temporal_envelope_for_query(self, query: str) -> Dict[str, Any]:
        """Discovers the temporal span and associated years for query entities."""
        entities = self.extract_entities(query)
        associated_years: Set[int] = set()

        for ent_name, _ in entities:
            ent_node = f"entity:{ent_name.lower()}"
            if self.graph.has_node(ent_node):
                for neighbor in self.graph.neighbors(ent_node):
                    if self.graph.nodes[neighbor].get("node_type") == "year":
                        associated_years.add(self.graph.nodes[neighbor]["value"])

        direct_years = re.findall(r"\b(19[789]\d|20[012]\d)\b", query)
        for y in direct_years:
            associated_years.add(int(y))

        min_year = min(associated_years) if associated_years else 1980
        max_year = max(associated_years) if associated_years else 2024
        return {
            "entities": [e[0] for e in entities],
            "years": sorted(list(associated_years)),
            "temporal_span": (min_year, max_year),
        }


# =====================================================================
# Epistemic Graph Traverser & Tacit HyDE Generator
# =====================================================================

class EpistemicGraphTraverser:
    """
    Traverses NetworkX epistemic transition edges to retrieve paired antecedent
    and successor evidence chunks whenever queries ask about scientific pivots.
    """
    SHIFT_KEYWORDS = [
        "move", "moved", "switch", "switched", "transition", "transitioned",
        "change mind", "changed my mind", "change my mind", "pivot", "pivoted",
        "instead of", "why did i", "why did we", "evolution", "supersede",
    ]

    def __init__(self, graph_extractor: TemporalGraphExtractor):
        self.graph_extractor = graph_extractor

    def is_shift_query(self, query: str) -> bool:
        """Determines if query is investigating a change in scientific belief or methodology."""
        lower_q = query.lower()
        return any(kw in lower_q for kw in self.SHIFT_KEYWORDS)

    def find_shift_grounding_chunks(self, query: str) -> List[str]:
        """
        Finds chunk IDs linked to transition edges relevant to query concepts.
        """
        if not self.is_shift_query(query):
            return []

        g = self.graph_extractor.graph
        entities = self.graph_extractor.extract_entities(query)
        shift_chunk_ids: List[str] = []

        for ent_name, _ in entities:
            node_id = f"entity:{ent_name.lower()}"
            if not g.has_node(node_id):
                continue

            # Outgoing shift edges
            for _, target, data in g.out_edges(node_id, data=True):
                if data.get("relation") in ["SUPERSEDED_BY", "CHANGED_MIND_ON", "EVOLVED_INTO"]:
                    cid = data.get("chunk_id")
                    if cid:
                        shift_chunk_ids.append(cid)

            # Incoming shift edges
            for source, _, data in g.in_edges(node_id, data=True):
                if data.get("relation") in ["SUPERSEDED_BY", "CHANGED_MIND_ON", "EVOLVED_INTO"]:
                    cid = data.get("chunk_id")
                    if cid:
                        shift_chunk_ids.append(cid)

        logger.info(f"Epistemic graph traversal discovered {len(shift_chunk_ids)} transition grounding chunks.")
        return list(dict.fromkeys(shift_chunk_ids))


class TacitHyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE) tailored to Michael Snyder's conversational voice.
    Converts 3rd-person questions into spoken 1st-person tacit intuition monologues.
    """
    # Deterministic domain voice templates for key historical epochs
    DOMAIN_REFLECTIONS = {
        "yeast": "In 1985 at Stanford and Yale, I decided to switch from Drosophila to yeast because fruit flies lacked DNA transformation at the time and yeast was a dream system where you could make a mutation in a gene and put it back.",
        "drosophila": "After graduate school at Caltech working on Drosophila cuticle genes with Norman Davidson, I wanted to be broad and learn a new organism and new techniques, which led to the transition to yeast with Ron Davis.",
        "microarray": "In the late 1990s, we realized single-gene studies were insufficient and developed DNA microarrays and ChIP-chip to profile whole-genome expression and transcription factor binding simultaneously.",
        "transposon": "We built transposon mutagenesis libraries in yeast to systematically mutagenize and epitope-tag every gene across the entire eukaryotic genome.",
        "ipop": "In 2010 at Stanford, we launched personal omics profiling (iPOP) because static DNA only tells you predisposition, whereas longitudinal multi-omics reveals real-time disease onset.",
        "continuous glucose": "Continuous glucose monitoring (CGM) showed us that people have distinct glucotypes and spike to different foods, shifting medicine from reactive sick-care to continuous preventative monitoring.",
        "wearable": "Smartwatches and continuous sensors measure physiological deviations continuously, allowing early detection of infections like Lyme disease and COVID-19 before clinical symptoms.",
        "norman davidson": "Norman Davidson had the best analytical mind and taught me that science is the most important issue, whereas Ron Davis was the ultimate creative genius who could brainstorm ten biological questions on the spot.",
        "ron davis": "Ron Davis had the most creative mind in science, inventing techniques and perceiving new biological questions, complementing Norman Davidson's deep analytical rigor.",
        "botstein": "Working with Ron Davis, David Botstein, and Norman Davidson shaped my entire philosophy toward developing scalable tools and technologies rather than chasing single-gene hypotheses.",
        "collaborator": "Early on, working with pioneers like Ron Davis, David Botstein, and Norman Davidson taught me to focus on building technologies that empower the entire scientific community.",
        "harvard": "When I was considering faculty positions in 1986, I was offered a job at both Harvard and Yale. I turned down Harvard because junior faculty were not treated well there, and after your fourth year you stop getting graduate students because they know you are not going to get tenure.",
        "politics of science": "Karl Drlica tried to teach me the politics of science and gave me Academic Gamesmanship, but my philosophy was always to meet people for the science they do rather than political networking for grants.",
        "politics": "Karl Drlica suggested I apply to hotshots and read Academic Gamesmanship, but I believed you should meet people only because you are interested in their science, and the grants and chips will fall into place.",
        "autoimmune": "When I was looking for autoimmune sera at Yale to clone mitotic spindle and kinetochore proteins, rheumatologists were very reluctant to share their clinical sera collections until Eric Lambie screened all 892 sera in a month and a half.",
        "sera": "When I first went looking for autoimmune sera, rheumatologists did not want to give up their clinical sera even though they were not using them, which taught me the contrast between clinical hoarding and molecular biology sharing.",
        "bench": "I have always had a visceral love of working at the bench with my own hands and making my own plasmids so my lab members do not have to do junk work, and I dislike administrative committee meetings.",
        "chemistry": "My high school chemistry teacher Mr. Darby ran a completely freelance, unstructured advanced class where we were left on our own to do experiments, which directly shaped the loose, independent way I run my own lab.",
        "darby": "Mr. Darby gave us independent lab experiments where we were on our own, which taught me self-direction and shaped the loose management style of my laboratory.",
        "stanford": "I was recruited from Yale to Stanford to become Chair of the Department of Genetics, build the premier systems biology and genomics program, and return to California.",
        "leave yale": "I left Yale for Stanford to serve as Chair of the Department of Genetics, build a world-class genomics and personalized medicine center, and return to California.",
        "profile a human": "When we decided to pioneer longitudinal personal multi-omics profiling (iPOP), I started with myself as subject zero because it was a high-risk proof of concept requiring hundreds of blood draws, which directly revealed my own type 2 diabetes onset.",
        "myself": "I started longitudinal multi-omics profiling on myself because profiling tens of thousands of molecules was unproven and required intense longitudinal sampling, which detected my viral infections and glucose spikes.",
        "wrong about": "In the 1980s and 1990s, the field believed disease was predominantly determined by static genetics, but our longitudinal multi-omics proved that dynamic environmental exposures, viruses, and the exposome play a massive role.",
        "competition": "Early in my career at Yale, science was competitive around reagents, but I came to believe strongly in open sharing of transposon libraries and microarrays so everyone could build on them.",
        "crowded": "I always avoided working in crowded, secretive areas like oncogenes, preferring to work in unique areas or collaborate with colleagues like Joel Rosenbaum and Phil Hieter to create best-fit scenarios.",
        "sharing": "I believed that sharing reagents openly accelerated the whole field. When we built the yeast transposon libraries, we distributed them freely to hundreds of labs.",
        "pick problem": "When I was starting out, my philosophy on picking problems was to tackle questions where existing methods failed, and invent new high-throughput technology to open up the problem.",
        "tenure": "During my early years at Yale, I encouraged students and postdocs to have independent projects they could take with them, fostering genuine scientific independence.",
        "mentoring": "During my early years at Yale, I encouraged students and postdocs to have independent projects they could take with them, fostering genuine scientific independence.",
        "exposome": "In the 1980s we focused almost purely on genetics, but over time I realized that environmental exposures and the personal exposome account for a massive fraction of disease risk.",
    }

    def generate_tacit_reflection(self, query: str) -> str:
        """Generates a first-person tacit intuition reflection matching Snyder's style."""
        lower_q = query.lower()
        matched_reflections = []

        for kw, text in self.DOMAIN_REFLECTIONS.items():
            if kw in lower_q:
                matched_reflections.append(text)

        if matched_reflections:
            return " ".join(matched_reflections)

        # General tacit spoken monologue framing
        return f"[Michael Snyder]: Over my career, my intuition was that high-throughput systems biology and unbiased measurement always beat narrow hypotheses. When asked '{query}', the key was capturing dynamic longitudinal data."


# =====================================================================
# BM25 Sparse Search & Reciprocal Rank Fusion (RRF)
# =====================================================================

class BM25SearchEngine:
    """
    Sparse keyword retrieval engine using BM25Okapi for precision technical term matching.
    """
    def __init__(self):
        self.chunk_ids: List[str] = []
        self.corpus_chunks: Dict[str, Dict[str, Any]] = {}
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None
        self._initialized = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Builds or rebuilds BM25 inverted index from a list of chunk dicts."""
        from rank_bm25 import BM25Okapi
        self.chunk_ids = []
        self.corpus_chunks = {}
        self.tokenized_corpus = []

        for c in chunks:
            cid = c.get("chunk_id") or c.get("id")
            if not cid:
                continue
            text = c.get("raw_text") or c.get("text") or ""
            self.chunk_ids.append(cid)
            self.corpus_chunks[cid] = c
            self.tokenized_corpus.append(self.tokenize(text))

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self._initialized = True
            logger.info(f"Indexed {len(self.chunk_ids)} chunks into BM25 sparse search engine.")

    def search(self, query: str, top_k: int = 40) -> List[Tuple[str, float]]:
        """Searches BM25 index returning (chunk_id, bm25_score) tuples."""
        if not self._initialized or not self.bm25 or not self.chunk_ids:
            return []
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def reciprocal_rank_fusion(
    dense_candidates: List[Dict[str, Any]],
    sparse_matches: List[Tuple[str, float]],
    chunk_lookup: Dict[str, Dict[str, Any]],
    dense_weight: float = settings.retrieval.bm25.dense_weight,
    sparse_weight: float = settings.retrieval.bm25.sparse_weight,
    rrf_k: int = settings.retrieval.bm25.rrf_k,
) -> List[Dict[str, Any]]:
    """
    Combines dense similarity ranking with sparse BM25 ranking via Reciprocal Rank Fusion:
    RRF(d) = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)
    """
    fused_scores: Dict[str, float] = defaultdict(float)
    fused_chunks: Dict[str, Dict[str, Any]] = {}

    # Dense ranks (rank is 0-indexed)
    for rank, cand in enumerate(dense_candidates):
        cid = cand["chunk_id"]
        fused_scores[cid] += dense_weight / (rrf_k + rank + 1)
        fused_chunks[cid] = cand

    # Sparse BM25 ranks
    for rank, (cid, bm_score) in enumerate(sparse_matches):
        fused_scores[cid] += sparse_weight / (rrf_k + rank + 1)
        if cid not in fused_chunks and cid in chunk_lookup:
            cand = chunk_lookup[cid].copy()
            # Normalize BM25 raw score to a pseudo similarity
            cand["similarity"] = min(1.0, bm_score / 15.0)
            fused_chunks[cid] = cand

    results = []
    for cid, score in fused_scores.items():
        if cid in fused_chunks:
            item = fused_chunks[cid].copy()
            item["rrf_score"] = score
            results.append(item)

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


# =====================================================================
# Local Cross-Encoder Re-ranker (RTX 5090 Acceleration)
# =====================================================================

class LocalCrossEncoderReranker:
    """
    Local Cross-Encoder re-ranking powered by RTX 5090 (or CPU fallback).
    Scores (query, chunk_text) pairs jointly to capture deep semantic relevance.
    """
    def __init__(
        self,
        model_name: str = settings.retrieval.reranker.model_name,
        enabled: bool = settings.retrieval.reranker.enabled,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self._model = None
        self._load_attempted = False

    def _load_model(self):
        if not self._load_attempted and self.enabled:
            self._load_attempted = True
            try:
                from sentence_transformers import CrossEncoder
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = CrossEncoder(self.model_name, device=device)
                logger.info(f"Loaded Local CrossEncoder ({self.model_name}) on device: {device}")
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder ({e}). Reranking will use RRF composite scores.")
                self._model = None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 30) -> List[Dict[str, Any]]:
        """Re-scores candidate chunks with the cross-encoder."""
        if not self.enabled or not candidates:
            return candidates

        self._load_model()
        if self._model is None:
            return candidates

        rerank_slice = candidates[:top_n]
        pairs = [(query, c.get("raw_text", "")) for c in rerank_slice]

        try:
            import numpy as np
            scores = self._model.predict(pairs)
            for i, score in enumerate(scores):
                # Sigmoid normalization so cross-encoder is strictly [0.0, 1.0]
                prob = float(1.0 / (1.0 + np.exp(-float(score))))
                rerank_slice[i]["cross_encoder_score"] = prob
                # Combine Cross-Encoder with Source Tier boost
                tier = int(rerank_slice[i].get("tier", 1))
                tier_boost = {1: 0.25, 2: 0.15, 3: 0.05, 4: 0.0}.get(tier, 0.0)
                rerank_slice[i]["composite_score"] = prob + tier_boost

            reranked_top = sorted(rerank_slice, key=lambda x: x.get("composite_score", 0.0), reverse=True)
            return reranked_top + candidates[top_n:]
        except Exception as e:
            logger.warning(f"CrossEncoder prediction error ({e}). Returning un-reranked candidates.")
            return candidates


# =====================================================================
# Anti-Flattening Chronological Retriever
# =====================================================================

class AntiFlatteningRetriever:
    """
    RAG Retriever that combines Hybrid BM25 + Dense RRF Search, Tacit HyDE Expansion,
    Epistemic Graph Shift Traversal, Local Cross-Encoder re-ranking, and chronological
    epoch stratification (capped at max 15 chunks).
    """
    def __init__(
        self,
        chroma_store: ChromaStore,
        embed_engine: LocalEmbeddingEngine,
        graph_extractor: Optional[TemporalGraphExtractor] = None,
        max_chunks: int = settings.retrieval.max_llm_chunks,
    ):
        self.chroma_store = chroma_store
        self.embed_engine = embed_engine
        self.graph_extractor = graph_extractor or TemporalGraphExtractor()
        self.bm25_engine = BM25SearchEngine()
        self.reranker = LocalCrossEncoderReranker()
        self.hyde_generator = TacitHyDEGenerator()
        self.graph_traverser = EpistemicGraphTraverser(self.graph_extractor)
        self.max_chunks = max_chunks
        self._warm_bm25()

    def _warm_bm25(self):
        """Preloads BM25 index from existing ChromaDB chunks on startup."""
        all_chunks = self.chroma_store.load_all_chunks()
        if all_chunks:
            self.bm25_engine.index_chunks(all_chunks)

    def refresh_indices(self):
        """Refreshes BM25 corpus after new documents are ingested."""
        self._warm_bm25()

    def retrieve(self, query: str, top_k_candidates: int = 40) -> List[Dict[str, Any]]:
        """
        Executes the high-precision retrieval funnel:
        1. Graph Temporal Envelope Discovery & Epistemic Shift Edge Traversal
        2. Tacit HyDE First-Person Query Expansion
        3. Hybrid Dense Vector (ChromaDB) + Sparse Lexical (BM25) RRF Fusion
        4. Local Cross-Encoder Re-ranking
        5. Stratified Chronological Epoch Allocation (Max 15 Chunks)
        """
        logger.info(f"Retrieving for query: '{query}'")

        # Step 1: Graph Query & Epistemic Shift Traversal
        envelope = self.graph_extractor.get_temporal_envelope_for_query(query)
        logger.info(f"Graph temporal envelope: entities={envelope['entities']}, span={envelope['temporal_span']}")
        shift_chunk_ids = self.graph_traverser.find_shift_grounding_chunks(query)

        # Step 2A: Dense Semantic Search via ChromaDB with Tacit HyDE Expansion
        query_emb = self.embed_engine.embed_query(query)
        dense_candidates_raw = self.chroma_store.query(query_emb, n_results=top_k_candidates)

        # Tacit HyDE First-Person Reflection Embedding
        hyde_text = self.hyde_generator.generate_tacit_reflection(query)
        hyde_emb = self.embed_engine.embed_query(hyde_text)
        hyde_candidates = self.chroma_store.query(hyde_emb, n_results=top_k_candidates // 2)

        # Merge dense candidates
        dense_candidates_map = {c["chunk_id"]: c for c in dense_candidates_raw}
        for hc in hyde_candidates:
            cid = hc["chunk_id"]
            if cid not in dense_candidates_map:
                dense_candidates_map[cid] = hc
            else:
                dense_candidates_map[cid]["similarity"] = max(dense_candidates_map[cid]["similarity"], hc["similarity"])

        dense_candidates = sorted(dense_candidates_map.values(), key=lambda x: x["similarity"], reverse=True)

        # Step 2B: Sparse BM25 Search
        if not self.bm25_engine._initialized:
            self._warm_bm25()

        sparse_matches = self.bm25_engine.search(query, top_k=top_k_candidates)

        # Step 2C: Reciprocal Rank Fusion (RRF)
        all_chunks_dict = {c["chunk_id"]: c for c in self.chroma_store.load_all_chunks()}
        fused_candidates = reciprocal_rank_fusion(
            dense_candidates=dense_candidates,
            sparse_matches=sparse_matches,
            chunk_lookup=all_chunks_dict,
        )

        if not fused_candidates:
            logger.warning("No candidate chunks found across dense and sparse search.")
            return []

        # Populate year, tier, and shift boost for scoring
        for cand in fused_candidates:
            meta = cand.get("metadata", {})
            year = int(meta.get("year", 1991))
            tier = int(meta.get("tier", 1))
            cand["year"] = year
            cand["tier"] = tier

            tier_boost = {1: 0.25, 2: 0.15, 3: 0.05, 4: 0.0}.get(tier, 0.0)
            # Epistemic shift grounding chunk boost (+0.35)
            shift_boost = 0.35 if cand["chunk_id"] in shift_chunk_ids else 0.0
            cand["composite_score"] = cand.get("rrf_score", 0.0) + tier_boost + shift_boost

        # Step 3: Local Cross-Encoder Re-ranking
        reranked_candidates = self.reranker.rerank(
            query=query,
            candidates=fused_candidates,
            top_n=settings.retrieval.reranker.top_n_rerank,
        )

        # Step 4: Chronological Stratification Across 4 Historical Eras
        era_bins: Dict[str, List[Dict[str, Any]]] = {
            "early_career_1975_1989": [],
            "genomics_pioneering_1990_2005": [],
            "personal_omics_2006_2015": [],
            "precision_health_2016_present": [],
            "undated_or_general": [],
        }

        for cand in reranked_candidates:
            year = cand["year"]
            if year <= 1989:
                era_bins["early_career_1975_1989"].append(cand)
            elif 1990 <= year <= 2005:
                era_bins["genomics_pioneering_1990_2005"].append(cand)
            elif 2006 <= year <= 2015:
                era_bins["personal_omics_2006_2015"].append(cand)
            elif year >= 2016:
                era_bins["precision_health_2016_present"].append(cand)
            else:
                era_bins["undated_or_general"].append(cand)

        # Sort each bin by composite score descending
        for b in era_bins:
            era_bins[b].sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)

        # Stratified allocation: prioritize top global matches + ensure historical era breadth
        selected_chunks: List[Dict[str, Any]] = []
        selected_ids: Set[str] = set()

        all_sorted = sorted(reranked_candidates, key=lambda x: x.get("composite_score", 0.0), reverse=True)

        # Pass 1: Guarantee the top 6 highest-scoring global candidates
        for item in all_sorted[:6]:
            if item["chunk_id"] not in selected_ids and len(selected_chunks) < self.max_chunks:
                selected_chunks.append(item)
                selected_ids.add(item["chunk_id"])

        # Pass 2: Pick top 2 from every non-empty era to prevent temporal flattening
        for b, items in era_bins.items():
            for item in items[:2]:
                if item["chunk_id"] not in selected_ids and len(selected_chunks) < self.max_chunks:
                    selected_chunks.append(item)
                    selected_ids.add(item["chunk_id"])

        # Pass 3: Fill remaining capacity with next best scoring candidates
        for item in all_sorted:
            if len(selected_chunks) >= self.max_chunks:
                break
            if item["chunk_id"] not in selected_ids:
                selected_chunks.append(item)
                selected_ids.add(item["chunk_id"])

        # Crucial: Sort final selected chunks in chronological order so Claude sees the evolution
        selected_chunks.sort(key=lambda x: (x["year"], -x.get("composite_score", 0.0)))

        logger.info(f"Selected {len(selected_chunks)} stratified chronological chunks across years: {[c['year'] for c in selected_chunks]}")
        return selected_chunks
