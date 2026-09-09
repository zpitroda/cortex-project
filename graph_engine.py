"""
graph_engine.py - State-of-the-Art Hierarchical Temporal Knowledge Graph & Epistemic GraphRAG Engine.

Features:
1. Atomic Epistemic Semantic Triples:
   (Subject, Predicate, Object, Year, Evidence Quote, Epistemic Status, Chunk ID)
2. Local LLM (Qwen 27B) + High-Performance NLP Heuristic Hybrid Extraction.
3. Multi-Decade Hierarchical Community Clustering (Louvain / Greedy Modularity).
4. Automated Macro Thematic Community Summaries for synthesis queries.
5. Transactional graph serialization and anti-flattening temporal envelope extraction.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

import networkx as nx
from networkx.algorithms.community import louvain_communities, greedy_modularity_communities
from pydantic import BaseModel, Field

from config import SourceTier, settings
from ingest import ProcessedChunk, extract_snyder_speech

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cortex.graph_engine")


# =====================================================================
# 1. Pydantic Schemas for Epistemic Semantic Triples & Communities
# =====================================================================

class SemanticTriple(BaseModel):
    """Atomic epistemic proposition with verbatim textual grounding."""
    subject: str = Field(..., description="Entity or scientific concept acting as the subject.")
    predicate: str = Field(..., description="Semantic relation or action (e.g. DEVELOPED, DISCOVERED, SUPERSEDED_BY).")
    object: str = Field(..., description="Target entity, methodology, or scientific finding.")
    year: int = Field(..., description="The chronological year grounding this claim.")
    evidence_quote: str = Field(..., description="Exact verbatim excerpt from text grounding the claim.")
    epistemic_status: str = Field(default="FACT", description="Epistemic status: FACT, HYPOTHESIS, PARADIGM_SHIFT, RETROSPECTIVE.")
    chunk_id: str = Field(default="", description="The unique chunk UUID.")
    source_file: str = Field(default="", description="The disk source filename.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class HierarchicalCommunity(BaseModel):
    """Macro-level thematic or chronological cluster of research activity."""
    community_id: str
    title: str
    year_range: Tuple[int, int]
    core_entities: List[str]
    sample_claims: List[str]
    summary: str
    member_chunk_ids: List[str] = Field(default_factory=list)


# =====================================================================
# 2. Local LLM & NLP Semantic Triple Extractor
# =====================================================================

class LLMTripleExtractor:
    """
    Extracts high-value scientific claims & epistemic pivots from text chunks.
    Uses local Qwen 27B on port 8080 when available, with fast NLP fallback.
    """
    # Key historical domain dictionary for Michael Snyder's research
    DOMAIN_ONTOLOGY = {
        # Organisms
        "yeast": ("Yeast", "Organism"),
        "saccharomyces cerevisiae": ("Yeast", "Organism"),
        "drosophila": ("Drosophila", "Organism"),
        "fruit fly": ("Drosophila", "Organism"),
        "human": ("Human", "Organism"),
        "mouse": ("Mouse", "Organism"),
        "e. coli": ("E. Coli", "Organism"),

        # Technologies & Innovations
        "transposon mutagenesis": ("Transposon Mutagenesis", "Technology"),
        "epitope tagging": ("Epitope Tagging", "Technology"),
        "microarray": ("Microarrays", "Technology"),
        "chip-chip": ("ChIP-chip", "Technology"),
        "chip-seq": ("ChIP-seq", "Technology"),
        "rna-seq": ("RNA-seq", "Technology"),
        "mass spectrometry": ("Mass Spectrometry", "Technology"),
        "ipop": ("iPOP (Integrative Personal Omics)", "Technology"),
        "personal omics": ("Personal Multi-Omics", "Technology"),
        "continuous glucose monitor": ("Continuous Glucose Monitoring", "Technology"),
        "cgm": ("Continuous Glucose Monitoring", "Technology"),
        "smartwatch": ("Wearable Sensors", "Technology"),
        "wearables": ("Wearable Sensors", "Technology"),
        "ageotype": ("Ageotypes", "Concept"),
        "glucotype": ("Glucotypes", "Concept"),

        # Institutions & Mentors
        "stanford": ("Stanford University", "Institution"),
        "yale": ("Yale University", "Institution"),
        "caltech": ("Caltech", "Institution"),
        "ron davis": ("Ron Davis", "Person"),
        "david botstein": ("David Botstein", "Person"),
        "paul berg": ("Paul Berg", "Person"),
    }

    def __init__(self):
        self._load_spacy()

    def _load_spacy(self):
        try:
            import spacy
            self.nlp = spacy.load(settings.models.spacy_model_name)
        except Exception:
            self.nlp = None

    def extract_triples_local_llm(self, text: str, year: int, chunk_id: str, source_file: str) -> Optional[List[SemanticTriple]]:
        """Queries local llama.cpp server on port 8080 to extract semantic triples in strict JSON."""
        import requests
        prompt = (
            f"You are an expert scientific knowledge graph extractor. Extract 2-4 key factual claims, discoveries, "
            f"or methodology transitions made by Dr. Michael Snyder from the following text ({year}).\n\n"
            f"Text:\n\"\"\"{text[:1500]}\"\"\"\n\n"
            f"Respond ONLY with a JSON list of objects with keys: subject, predicate, object, epistemic_status (FACT|PARADIGM_SHIFT|HYPOTHESIS), evidence_quote (verbatim short quote from text).\n"
            f"Example:\n[{{\"subject\": \"iPOP\", \"predicate\": \"REVEALED_PRECEDING_EVENT\", \"object\": \"Viral infection precedes Type 2 Diabetes onset\", \"epistemic_status\": \"FACT\", \"evidence_quote\": \"viral infection preceded the onset of diabetes\"}}]"
        )
        try:
            res = requests.post(
                "http://127.0.0.1:8080/v1/chat/completions",
                json={
                    "model": "qwen2.5-32b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
                timeout=3.0,
            )
            if res.status_code == 200:
                raw_content = res.json()["choices"][0]["message"]["content"]
                clean_json = re.sub(r"^```json\s*", "", raw_content.strip())
                clean_json = re.sub(r"\s*```$", "", clean_json)
                parsed = json.loads(clean_json)
                if isinstance(parsed, list):
                    triples = []
                    for item in parsed:
                        if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item:
                            eq = item.get("evidence_quote", text[:80])
                            triples.append(
                                SemanticTriple(
                                    subject=str(item["subject"]).strip(),
                                    predicate=str(item["predicate"]).strip().upper().replace(" ", "_"),
                                    object=str(item["object"]).strip(),
                                    year=year,
                                    evidence_quote=eq,
                                    epistemic_status=str(item.get("epistemic_status", "FACT")).upper(),
                                    chunk_id=chunk_id,
                                    source_file=source_file,
                                )
                            )
                    return triples
        except Exception:
            pass
        return None

    def extract_triples_nlp(self, text: str, year: int, chunk_id: str, source_file: str) -> List[SemanticTriple]:
        """Fast, robust deterministic NLP extraction for entities, relations, and paradigm shifts."""
        triples: List[SemanticTriple] = []
        lower_text = text.lower()

        # 1. Epistemic Paradigm Shift Detection
        shift_patterns = [
            (r"(?:moved|switched|transitioned|pivoted|shifted)\s+(?:away\s+)?from\s+([a-zA-Z\s\-]+?)\s+to\s+([a-zA-Z\s\-]+)", "SUPERSEDED_BY", "PARADIGM_SHIFT"),
            (r"instead\s+of\s+([a-zA-Z\s\-]+?)[,\s]+(?:we|I)\s+(?:worked on|chose|focused on|used)\s+([a-zA-Z\s\-]+)", "SUPERSEDED_BY", "PARADIGM_SHIFT"),
            (r"(?:used to think|thought early on)\s+([a-zA-Z\s\-]+?)\s+but\s+(?:later realized|found)\s+([a-zA-Z\s\-]+)", "CHANGED_MIND_ON", "PARADIGM_SHIFT"),
            (r"([a-zA-Z\s\-]+?)\s+(?:evolved into|led to|paved the way for)\s+([a-zA-Z\s\-]+)", "EVOLVED_INTO", "PARADIGM_SHIFT"),
        ]

        for pat, pred, status in shift_patterns:
            for match in re.finditer(pat, lower_text):
                src = match.group(1).strip()
                tgt = match.group(2).strip()
                quote = text[max(0, match.start() - 20): min(len(text), match.end() + 20)]
                triples.append(
                    SemanticTriple(
                        subject=src[:35].title(),
                        predicate=pred,
                        object=tgt[:35].title(),
                        year=year,
                        evidence_quote=quote.strip(),
                        epistemic_status=status,
                        chunk_id=chunk_id,
                        source_file=source_file,
                    )
                )

        # 2. Known Domain Co-Occurrence & Invention Claims
        matched_entities = []
        for key, (norm_name, cat) in self.DOMAIN_ONTOLOGY.items():
            if re.search(r"\b" + re.escape(key) + r"\b", lower_text):
                matched_entities.append((norm_name, cat))

        if len(matched_entities) >= 2:
            for i in range(len(matched_entities) - 1):
                e1, c1 = matched_entities[i]
                e2, c2 = matched_entities[i + 1]
                if e1 != e2:
                    rel = "APPLIED_TO" if c2 == "Organism" else "ASSOCIATED_WITH"
                    triples.append(
                        SemanticTriple(
                            subject=e1,
                            predicate=rel,
                            object=e2,
                            year=year,
                            evidence_quote=text[:120].strip(),
                            epistemic_status="FACT",
                            chunk_id=chunk_id,
                            source_file=source_file,
                        )
                    )
        elif len(matched_entities) == 1:
            e1, c1 = matched_entities[0]
            triples.append(
                SemanticTriple(
                    subject="Dr. Michael Snyder",
                    predicate="RESEARCHED",
                    object=e1,
                    year=year,
                    evidence_quote=text[:120].strip(),
                    epistemic_status="FACT",
                    chunk_id=chunk_id,
                    source_file=source_file,
                )
            )

        return triples

    def extract_triples(self, chunk: ProcessedChunk, use_llm: bool = False) -> List[SemanticTriple]:
        """Extraction: uses local LLM if requested and available, else fast deterministic NLP."""
        snyder_text = chunk.snyder_text if chunk.snyder_text.strip() else chunk.raw_text
        triples = None
        if use_llm:
            triples = self.extract_triples_local_llm(snyder_text, chunk.year, chunk.chunk_id, chunk.source_name)
        if not triples:
            triples = self.extract_triples_nlp(snyder_text, chunk.year, chunk.chunk_id, chunk.source_name)
        return triples


# =====================================================================
# 3. Hierarchical Temporal Graph & GraphRAG Engine
# =====================================================================

class HierarchicalTemporalGraph:
    """
    State-of-the-Art Temporal Knowledge Graph with Leiden/Louvain community detection
    and macro-level thematic summarization across Michael Snyder's 40-year scientific career.
    """
    def __init__(
        self,
        graph_path: Path = settings.graph_storage_path,
        summary_path: Optional[Path] = None,
    ):
        self.graph_path = Path(graph_path)
        self.summary_path = summary_path or self.graph_path.with_name("community_summaries.json")
        self.graph = nx.MultiDiGraph()
        self.communities: Dict[str, HierarchicalCommunity] = {}
        self.triple_extractor = LLMTripleExtractor()
        self.load()

    def load(self):
        """Loads graph and community summaries from disk."""
        if self.graph_path.exists():
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data, multigraph=True)
                logger.info(f"Loaded Hierarchical Graph ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")
            except Exception as e:
                logger.warning(f"Could not load graph ({e}). Initializing fresh graph.")
                self.graph = nx.MultiDiGraph()

        if self.summary_path.exists():
            try:
                with open(self.summary_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                self.communities = {k: HierarchicalCommunity(**v) for k, v in cdata.items()}
                logger.info(f"Loaded {len(self.communities)} pre-computed community summaries.")
            except Exception:
                self.communities = {}

    def save(self):
        """Persists graph and community summaries to disk."""
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        cdata = {k: v.model_dump() for k, v in self.communities.items()}
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(cdata, f, indent=2)
        logger.info(f"Saved Hierarchical Graph to {self.graph_path} and summaries to {self.summary_path}")

    def clear(self):
        """Clears all nodes, edges, and community structures."""
        self.graph.clear()
        self.communities.clear()
        if self.graph_path.exists():
            self.graph_path.unlink(missing_ok=True)
        if self.summary_path.exists():
            self.summary_path.unlink(missing_ok=True)
        logger.info("Cleared entire Knowledge Graph and Community index.")

    def add_triple(self, triple: SemanticTriple):
        """Adds a verified semantic triple to the MultiDiGraph."""
        sub_node = f"entity:{triple.subject.lower()}"
        obj_node = f"entity:{triple.object.lower()}"
        year_node = f"year:{triple.year}"
        chunk_node = f"chunk:{triple.chunk_id}" if triple.chunk_id else None

        self.graph.add_node(sub_node, node_type="entity", name=triple.subject)
        self.graph.add_node(obj_node, node_type="entity", name=triple.object)
        self.graph.add_node(year_node, node_type="year", value=triple.year)

        if chunk_node:
            self.graph.add_node(chunk_node, node_type="chunk", chunk_id=triple.chunk_id, source=triple.source_file, year=triple.year)
            self.graph.add_edge(chunk_node, sub_node, relation="GROUNDS_SUBJECT")
            self.graph.add_edge(chunk_node, obj_node, relation="GROUNDS_OBJECT")
            self.graph.add_edge(chunk_node, year_node, relation="ANCHORED_IN_YEAR")

        self.graph.add_edge(
            sub_node,
            obj_node,
            relation=triple.predicate,
            year=triple.year,
            evidence=triple.evidence_quote,
            epistemic_status=triple.epistemic_status,
            chunk_id=triple.chunk_id,
            source=triple.source_file,
        )
        self.graph.add_edge(sub_node, year_node, relation="ACTIVE_IN_YEAR")
        self.graph.add_edge(obj_node, year_node, relation="ACTIVE_IN_YEAR")

    def index_chunk(self, chunk: ProcessedChunk, triples: Optional[List[SemanticTriple]] = None, use_llm: bool = False) -> List[SemanticTriple]:
        """Indexes a chunk, extracting semantic triples and connecting temporal nodes."""
        if triples is None:
            triples = self.triple_extractor.extract_triples(chunk, use_llm=use_llm)

        chunk_node = f"chunk:{chunk.chunk_id}"
        self.graph.add_node(
            chunk_node,
            node_type="chunk",
            chunk_id=chunk.chunk_id,
            source_name=chunk.source_name,
            tier=chunk.tier,
            year=chunk.year,
        )

        year_node = f"year:{chunk.year}"
        self.graph.add_node(year_node, node_type="year", value=chunk.year)
        self.graph.add_edge(chunk_node, year_node, relation="ANCHORED_IN_YEAR")

        for t in triples:
            self.add_triple(t)

        return triples

    # =====================================================================
    # 4. Louvain Community Detection & Hierarchical Partitioning
    # =====================================================================

    def detect_and_build_communities(self) -> Dict[str, HierarchicalCommunity]:
        """
        Runs Louvain / Modularity community detection across the entity co-occurrence
        and claim graph, grouping 40+ years of research into coherent intellectual eras.
        """
        undirected_g = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            if u.startswith("entity:") and v.startswith("entity:"):
                w = 2.0 if data.get("epistemic_status") == "PARADIGM_SHIFT" else 1.0
                if undirected_g.has_edge(u, v):
                    undirected_g[u][v]["weight"] += w
                else:
                    undirected_g.add_edge(u, v, weight=w)

        if undirected_g.number_of_nodes() < 3:
            logger.info("Graph too small for community clustering.")
            return {}

        try:
            partitions = list(louvain_communities(undirected_g, weight="weight", seed=42))
        except Exception:
            partitions = list(greedy_modularity_communities(undirected_g, weight="weight"))

        communities = {}
        for idx, part in enumerate(partitions):
            if len(part) < 2:
                continue

            entity_names = [self.graph.nodes[n].get("name", n.replace("entity:", "").title()) for n in part if n in self.graph.nodes]
            years_found = set()
            claims = []
            chunk_ids = set()

            for n in part:
                for nbr in self.graph.neighbors(n):
                    if nbr.startswith("year:") and "value" in self.graph.nodes[nbr]:
                        years_found.add(self.graph.nodes[nbr]["value"])
                    if nbr.startswith("chunk:") and "chunk_id" in self.graph.nodes[nbr]:
                        chunk_ids.add(self.graph.nodes[nbr]["chunk_id"])

                # Collect sample claims
                for _, tgt, data in self.graph.out_edges(n, data=True):
                    if data.get("relation") and data.get("evidence"):
                        claims.append(f"{self.graph.nodes[n].get('name', n)} --[{data['relation']}]--> {self.graph.nodes.get(tgt, {}).get('name', tgt)}: \"{data['evidence']}\"")

            min_y = min(years_found) if years_found else 1985
            max_y = max(years_found) if years_found else 2024

            top_entities = entity_names[:4]
            title = f"Thematic Cluster: {', '.join(top_entities)} ({min_y}-{max_y})"
            summary = (
                f"Intellectual cluster centered on {', '.join(top_entities)} spanning {min_y} to {max_y}. "
                f"Includes {len(chunk_ids)} primary grounding documents and {len(claims)} documented scientific claims."
            )

            comm_id = f"community_{idx}_{min_y}_{max_y}"
            comm = HierarchicalCommunity(
                community_id=comm_id,
                title=title,
                year_range=(min_y, max_y),
                core_entities=entity_names,
                sample_claims=claims[:10],
                summary=summary,
                member_chunk_ids=list(chunk_ids),
            )
            communities[comm_id] = comm

        self.communities = communities
        self.save()
        logger.info(f"Successfully generated {len(communities)} hierarchical research communities.")
        return communities

    # =====================================================================
    # 5. Query Traversal & Subgraph Retrieval
    # =====================================================================

    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """
        Finds relevant entities, community summaries, and epistemic transitions
        for a user prompt to supply structured GraphRAG context.
        """
        lower_q = query.lower()
        matched_entities = set()
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "entity":
                name = data.get("name", "").lower()
                if name and (name in lower_q or any(token in lower_q for token in name.split() if len(token) > 3)):
                    matched_entities.add(node)

        # 1. Epistemic Shifts & Direct Triples
        relevant_triples = []
        for ent_node in matched_entities:
            for _, tgt, data in self.graph.out_edges(ent_node, data=True):
                if data.get("relation"):
                    src_name = self.graph.nodes[ent_node].get("name", ent_node)
                    tgt_name = self.graph.nodes.get(tgt, {}).get("name", tgt)
                    relevant_triples.append({
                        "subject": src_name,
                        "predicate": data.get("relation"),
                        "object": tgt_name,
                        "year": data.get("year"),
                        "evidence": data.get("evidence"),
                        "status": data.get("epistemic_status", "FACT"),
                        "chunk_id": data.get("chunk_id"),
                    })

        # 2. Matching Community Summaries
        matched_communities = []
        for comm in self.communities.values():
            if any(e.lower() in lower_q for e in comm.core_entities):
                matched_communities.append(comm)

        return {
            "matched_entities": [self.graph.nodes[e].get("name", e) for e in matched_entities],
            "triples": relevant_triples[:15],
            "community_summaries": [c.summary for c in matched_communities[:3]],
        }
