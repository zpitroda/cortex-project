"""
confidence.py - Pure Python Mathematical Confidence Engine.

Formula:
    Score = Base + Corroboration + Verification - Penalties

Rules of Evidence:
- Base (Max 35): Tier 1 (+35), Tier 2 (+30), Tier 3 (+15), Tier 4 (+5).
- Corroboration (Max 30): Sources span multiple years (+30), Multiple sources same year (+20), Single source (+10).
- Verification (Max 25): Exact substring search of citations' 'exact_quote' against raw chunk text.
    Crucial: Only searches text trailing the '[Michael Snyder]:' tag.
    Perfect match = +25. Partial match = +15. Hallucination/No match = 0.
- Penalties:
    is_inferred: true -> subtract 50 points.
    Verification == 0 -> Hard stop (Score = 0).
- Final score strictly bounded in [0, 100]%.
"""

import re
import difflib
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from config import SourceTier, LLMResponseSchema, settings
from ingest import extract_snyder_speech, extract_interviewer_speech

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cortex.confidence")


class CitationVerificationResult(BaseModel):
    chunk_id: str
    exact_quote: str
    found_exact: bool
    found_partial: bool
    similarity_ratio: float
    score_earned: float
    snyder_excerpt: str


class ConfidenceReport(BaseModel):
    """Complete mathematical breakdown of the calculated confidence score."""
    final_score: float = Field(..., description="Overall confidence score [0-100]%.")
    base_score: float = Field(..., description="Epistemological source tier score (Max 35).")
    corroboration_score: float = Field(..., description="Chronological and multi-source corroboration (Max 30).")
    recency_score: float = Field(..., description="Recency of sources (Max 10).")
    verification_score: float = Field(..., description="Direct quote substring verification score (Max 25).")
    penalty_score: float = Field(..., description="Inference and thin record penalties.")
    highest_tier: Optional[int] = None
    cited_years: List[int] = Field(default_factory=list)
    citations_verified: List[CitationVerificationResult] = Field(default_factory=list)
    is_inferred: bool = False
    thin_record: bool = False
    mathematical_rationale: str = ""


class MathematicalConfidenceEngine:
    """
    Deterministic confidence calculation engine.
    The LLM never asserts its own confidence; Python derives it mathematically.
    """
    def __init__(self, config=settings.confidence):
        self.config = config

    def calculate_base_score(self, cited_chunks: List[Dict[str, Any]]) -> Tuple[float, Optional[int]]:
        """
        Base (Max 35):
        Tier 1 source used: +35
        Tier 2: +30
        Tier 3: +15
        Tier 4: +5
        """
        if not cited_chunks:
            return 0.0, None

        tiers = [int(c.get("tier", c.get("metadata", {}).get("tier", 4))) for c in cited_chunks]
        min_tier = min(tiers)  # In SourceTier, 1 is highest priority

        if min_tier == SourceTier.TIER_1_ORAL_INTERVIEW:
            return self.config.tier_1_base, min_tier
        elif min_tier == SourceTier.TIER_2_PREPARED_LECTURE:
            return self.config.tier_2_base, min_tier
        elif min_tier == SourceTier.TIER_3_PUBLISHED_PAPER:
            return self.config.tier_3_base, min_tier
        elif min_tier == SourceTier.TIER_4_THIRD_PARTY:
            return self.config.tier_4_base, min_tier
        return 0.0, None

    def calculate_recency_score(self, cited_chunks: List[Dict[str, Any]]) -> float:
        """
        Recency (Max 10):
        At least one source >= 2010: +10
        At least one source >= 2000: +5
        All sources < 2000: +0
        """
        if not cited_chunks:
            return 0.0

        years = [int(c.get("year", c.get("metadata", {}).get("year", 1991))) for c in cited_chunks]
        max_year = max(years)

        if max_year >= 2010:
            return self.config.recency_post_2010
        elif max_year >= 2000:
            return self.config.recency_2000_2009
        return self.config.recency_pre_2000

    def calculate_corroboration_score(self, cited_chunks: List[Dict[str, Any]]) -> Tuple[float, List[int]]:
        """
        Corroboration (Max 30):
        Sources span multiple years: +30
        Multiple sources same year: +20
        Single source: +10
        0 sources: +0
        """
        if not cited_chunks:
            return 0.0, []

        years = []
        source_ids = set()
        for c in cited_chunks:
            y = int(c.get("year", c.get("metadata", {}).get("year", 1991)))
            sid = c.get("source_id", c.get("metadata", {}).get("source_id", c.get("chunk_id")))
            years.append(y)
            source_ids.add(sid)

        distinct_years = set(years)

        if len(distinct_years) > 1:
            return self.config.multi_year_corroboration, sorted(list(distinct_years))
        elif len(source_ids) > 1:
            return self.config.same_year_multi_source, sorted(list(distinct_years))
        elif len(cited_chunks) >= 1:
            return self.config.single_source, sorted(list(distinct_years))
        return 0.0, []

    def verify_exact_quote(self, exact_quote: str, raw_chunk_text: str) -> CitationVerificationResult:
        """
        Verification (Max 25):
        Executes .find() of exact_quote against only text trailing '[Michael Snyder]:'.
        Perfect match = +25.
        Partial match (>=82% sequential sliding-window match) = +12.5 (or +25 if >=92%).
        Interviewer speech match / Hallucination / No match = 0.
        """
        # Step 1: Forensically isolate text trailing [Michael Snyder]:
        snyder_text = extract_snyder_speech(raw_chunk_text)
        interviewer_text = extract_interviewer_speech(raw_chunk_text)

        def _clean_unicode_str(s: str) -> str:
            s = s.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
            s = s.replace("—", "-").replace("–", "-").replace("−", "-").replace("…", "...")
            s = re.sub(r"\s+", " ", s.strip().lower())
            return s

        norm_quote = _clean_unicode_str(exact_quote)
        norm_snyder = _clean_unicode_str(snyder_text)
        norm_interviewer = _clean_unicode_str(interviewer_text)

        # Invariant 4: Reject interviewer speech immediately
        # If the quote appears in interviewer speech and not in Snyder's speech, it is strictly rejected
        if norm_interviewer and norm_interviewer.find(norm_quote) != -1:
            if not norm_snyder or norm_snyder.find(norm_quote) == -1:
                return CitationVerificationResult(
                    chunk_id="",
                    exact_quote=exact_quote,
                    found_exact=False,
                    found_partial=False,
                    similarity_ratio=0.0,
                    score_earned=self.config.unverified_quote,
                    snyder_excerpt="(REJECTED: Quote was spoken by Interviewer, not Michael Snyder)",
                )

        if not norm_snyder:
            return CitationVerificationResult(
                chunk_id="",
                exact_quote=exact_quote,
                found_exact=False,
                found_partial=False,
                similarity_ratio=0.0,
                score_earned=self.config.unverified_quote,
                snyder_excerpt="(No Michael Snyder speech found in chunk)",
            )

        # Exact substring search (.find())
        if norm_snyder.find(norm_quote) != -1:
            return CitationVerificationResult(
                chunk_id="",
                exact_quote=exact_quote,
                found_exact=True,
                found_partial=True,
                similarity_ratio=1.0,
                score_earned=self.config.exact_quote_match,
                snyder_excerpt=snyder_text[:200],
            )

        # Ellipsis-separated exact sub-clause matching (academic quote omission convention)
        if "..." in norm_quote:
            sub_clauses = [c.strip() for c in norm_quote.split("...") if len(c.strip()) >= 6]
            if sub_clauses and all(norm_snyder.find(c) != -1 for c in sub_clauses):
                positions = [norm_snyder.find(c) for c in sub_clauses]
                if positions == sorted(positions):
                    return CitationVerificationResult(
                        chunk_id="",
                        exact_quote=exact_quote,
                        found_exact=True,
                        found_partial=True,
                        similarity_ratio=1.0,
                        score_earned=self.config.exact_quote_match,
                        snyder_excerpt=snyder_text[:200],
                    )

        # Partial matching via sequential sliding-window SequenceMatcher
        quote_words = norm_quote.split()
        snyder_words = norm_snyder.split()
        q_len = len(quote_words)
        best_ratio = 0.0

        if q_len > 0 and len(snyder_words) >= q_len:
            for w_len in (q_len, q_len - 1, q_len + 1, q_len + 2):
                if w_len <= 0 or w_len > len(snyder_words):
                    continue
                for i in range(len(snyder_words) - w_len + 1):
                    window = " ".join(snyder_words[i:i + w_len])
                    r = difflib.SequenceMatcher(None, norm_quote, window).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        if best_ratio >= 0.95:
                            break
                if best_ratio >= 0.95:
                    break
        elif q_len > 0 and len(snyder_words) > 0:
            best_ratio = difflib.SequenceMatcher(None, norm_quote, norm_snyder).ratio()

        if best_ratio >= 0.82:
            return CitationVerificationResult(
                chunk_id="",
                exact_quote=exact_quote,
                found_exact=False,
                found_partial=True,
                similarity_ratio=best_ratio,
                score_earned=self.config.exact_quote_match if best_ratio >= 0.92 else self.config.partial_quote_match,
                snyder_excerpt=snyder_text[:200],
            )

        # No match / Hallucination
        return CitationVerificationResult(
            chunk_id="",
            exact_quote=exact_quote,
            found_exact=False,
            found_partial=False,
            similarity_ratio=best_ratio,
            score_earned=self.config.unverified_quote,
            snyder_excerpt=snyder_text[:200],
        )

    def evaluate(
        self,
        llm_response: LLMResponseSchema,
        chunk_database: Dict[str, Dict[str, Any]],
    ) -> ConfidenceReport:
        """
        Calculates the complete deterministic confidence score for an LLM response.
        """
        citations = llm_response.citations
        flags = llm_response.flags

        # Resolve cited chunk objects
        cited_chunks = []
        verification_results: List[CitationVerificationResult] = []

        for cite in citations:
            chunk_data = chunk_database.get(cite.chunk_id)
            if chunk_data:
                cited_chunks.append(chunk_data)
                raw_text = chunk_data.get("raw_text", "")
                res = self.verify_exact_quote(cite.exact_quote, raw_text)
                res.chunk_id = cite.chunk_id
                verification_results.append(res)
            else:
                # Chunk ID not in database (fabricated ID)
                verification_results.append(CitationVerificationResult(
                    chunk_id=cite.chunk_id,
                    exact_quote=cite.exact_quote,
                    found_exact=False,
                    found_partial=False,
                    similarity_ratio=0.0,
                    score_earned=0.0,
                    snyder_excerpt="(Chunk ID not found in database)",
                ))

        # 1. Base Score (Max 35)
        base_score, highest_tier = self.calculate_base_score(cited_chunks)

        # 2. Corroboration Score (Max 30)
        corroboration_score, cited_years = self.calculate_corroboration_score(cited_chunks)

        # 3. Recency Score (Max 10)
        recency_score = self.calculate_recency_score(cited_chunks)

        # 4. Verification Score (Max 25)
        if verification_results:
            avg_verif = sum(v.score_earned for v in verification_results) / len(verification_results)
            verification_score = avg_verif
        else:
            verification_score = 0.0

        # 5. Penalties
        penalties = 0.0

        # Narrative thin-record heuristic detection
        narrative_lower = llm_response.narrative.lower()
        thin_narrative_indicators = [
            "does not contain direct documentation",
            "does not contain sufficient information",
            "cannot provide a definitive answer",
            "specific reasoning is not explicitly stated",
            "is not documented in these",
            "are not documented in these",
            "the record is silent",
            "none of the retrieved evidence discusses",
            "none of the retrieved evidence includes",
            "unclear from these retrieved documents",
            "does not contain information addressing the specific question",
            "remains undocumented in the available",
            "does not articulate",
            "is not captured in these particular retrieved",
            "reasoning for rejecting harvard specifically remains unclear",
        ]
        narrative_indicates_thin = any(ind in narrative_lower for ind in thin_narrative_indicators)

        is_inferred = bool(flags.is_inferred or narrative_indicates_thin)
        thin_record = bool(flags.thin_record or narrative_indicates_thin)

        if is_inferred:
            penalties += self.config.inference_penalty
        if thin_record:
            penalties += getattr(self.config, "thin_record_penalty", 40.0)

        # Unverified citations penalty: if any citation completely failed verification, penalize
        unverified_cits = [v for v in verification_results if v.score_earned == 0.0]
        if unverified_cits and citations:
            penalties += 15.0 * len(unverified_cits)

        # HARD RULE: If Verification == 0 (unverified or hallucinated quotes), final score is 0
        if verification_score == 0.0 and citations:
            final_score = 0.0
            rationale = "CRITICAL HARD STOP: Verification score is 0.0 (citations failed exact/partial verification against Michael Snyder's speech). Final confidence dropped to 0%."
        else:
            raw_score = base_score + corroboration_score + recency_score + verification_score - penalties
            final_score = max(0.0, min(100.0, raw_score))
            rationale = (
                f"Formula: Base({base_score:.1f}) + Corroboration({corroboration_score:.1f}) + "
                f"Recency({recency_score:.1f}) + Verification({verification_score:.1f}) - Penalties({penalties:.1f}) = {final_score:.1f}%."
            )

        report = ConfidenceReport(
            final_score=final_score,
            base_score=base_score,
            corroboration_score=corroboration_score,
            recency_score=recency_score,
            verification_score=verification_score,
            penalty_score=penalties,
            highest_tier=highest_tier,
            cited_years=cited_years,
            citations_verified=verification_results,
            is_inferred=is_inferred,
            thin_record=thin_record,
            mathematical_rationale=rationale,
        )

        logger.info(f"[Confidence Engine] Calculated Score: {final_score:.1f}% | Rationale: {rationale}")
        return report
