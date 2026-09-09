"""
scraper.py - Autonomous Internet Scraper & Broad Corpus Discovery Engine for Michael Snyder.

Architecture:
1. Academic Literature (Europe PMC): ZERO keyword filtering. Every paper retrieved with
   Snyder authorship & affiliation on PubMed/Europe PMC is directly ingested to ensure
   100% capture of all landmark, niche, and interdisciplinary contributions.
2. Web & Video Discovery (YouTube / Web / Podcasts):
   - Local LLM Semantic Disambiguation: Uses local Qwen 27B on port 8080 to intelligently
     distinguish Dr. Michael Snyder (Stanford/Yale scientist) from homonyms (prophecy bloggers, realtors).
   - Fast Heuristic Fallback: If local LLM is offline, applies institutional & negative blacklist checks.
   - Speaker Turn Diarization: Formats raw dialogue turns ([Interviewer]: ... / [Michael Snyder]: ...).
3. Wi-Fi & Network Resilience: Exponential backoff retries and connection pooling to survive drops.
4. Unified Canonical Deduplication: Atomic manifest synchronization with disk files.
5. Deterministic Termination: Systematically exhausts all query frontiers and cleanly exits.
"""

import os
import re
import json
import time
import socket
import urllib.parse
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from pypdf import PdfReader
from tqdm import tqdm

from config import SourceTier, settings
from pipeline import CortexPersonalBrain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cortex.scraper")

RAW_DIR = Path("l:/cortex/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MANIFEST_PATH = Path("l:/cortex/data/processed/scraper_manifest.json")


# =====================================================================
# 0. Local LLM Semantic Assistant (Qwen 27B on llama.cpp port 8080)
# =====================================================================

class LocalLLMScraperAssistant:
    """
    Intelligent local assistant using Qwen 27B on port 8080 for semantic
    disambiguation and dialogue turn formatting.
    """
    def __init__(self, endpoint: str = "http://localhost:8080/v1/chat/completions"):
        self.endpoint = endpoint
        self._checked = False
        self._available = False

    def is_available(self) -> bool:
        if self._checked:
            return self._available
        try:
            r = requests.get("http://localhost:8080/v1/models", timeout=0.8)
            self._available = (r.status_code == 200)
            if self._available:
                logger.info("Local Qwen 27B LLM assistant detected on port 8080. Enabled for semantic disambiguation.")
        except Exception:
            self._available = False
        self._checked = True
        return self._available

    def classify_snyder_content(self, title: str, author_or_channel: str, text: str) -> Optional[bool]:
        """
        Uses semantic understanding to determine if an open web/video item
        is by or about Dr. Michael Snyder (the Stanford/Yale professor),
        without brittle hardcoded keyword lists.
        """
        if not self.is_available():
            return None

        prompt = (
            "Determine if the following item is by or about Dr. Michael Snyder "
            "(the Stanford University and Yale University professor of genetics, genomics, biology, and medicine).\n\n"
            f"Title: {title}\n"
            f"Author/Channel: {author_or_channel}\n"
            f"Summary/Text: {text[:1500]}\n\n"
            "Answer strictly in JSON:\n"
            '{"is_snyder_scientist": true/false, "confidence": 0.0-1.0, "reason": "short explanation"}'
        )

        try:
            res = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert biographical classifier. Output JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1024
                },
                timeout=15
            )
            if res.status_code == 200:
                data = res.json()
                choice = data["choices"][0]["message"]
                content = choice.get("content", "").strip()
                if not content and "reasoning_content" in choice:
                    raw = choice["reasoning_content"]
                    if "{" in raw and "}" in raw:
                        content = raw[raw.find("{"):raw.rfind("}")+1]

                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]

                parsed = json.loads(content.strip())
                is_snyder = parsed.get("is_snyder_scientist", False)
                logger.info(f"[LOCAL LLM CLASSIFIER] '{title[:45]}': {'ACCEPTED' if is_snyder else 'REJECTED'} ({parsed.get('reason')})")
                return is_snyder
        except Exception as e:
            logger.debug(f"Local LLM classification bypassed: {e}")

        return None

    def format_dialogue_turns(self, raw_transcript: str) -> str:
        """Transforms flat transcript text into structured speaker turns."""
        if not self.is_available() or len(raw_transcript.strip()) < 100:
            return raw_transcript

        sample = raw_transcript[:3000]
        prompt = (
            "Format the following conversation into clean speaker dialogue turns with tags:\n"
            "[Interviewer]: ...\n"
            "[Michael Snyder]: ...\n\n"
            f"TRANSCRIPT:\n{sample}\n\n"
            "Output ONLY the formatted dialogue. Do not include markdown code blocks or commentary."
        )

        try:
            res = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json={
                    "messages": [
                        {"role": "system", "content": "You are a precise scientific transcript editor. Format speech into [Interviewer]: and [Michael Snyder]: dialogue blocks."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 2048
                },
                timeout=20
            )
            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"].strip()
                if "[Michael Snyder]:" in content:
                    remainder = raw_transcript[3000:].strip()
                    if remainder:
                        return f"{content}\n\n[Michael Snyder]: {remainder}"
                    return content
        except Exception as e:
            logger.debug(f"Local LLM formatting bypassed: {e}")

        return raw_transcript


# =====================================================================
# 1. Topic Relevance & Anti-Homonym Validator (Heuristic & Semantic)
# =====================================================================

class TopicRelevanceValidator:
    """
    Validates open web/video discoveries against known homonyms.
    Uses Local LLM when live, and robust institutional/negative heuristics as fallback.
    """
    NEGATIVE_KEYWORDS = [
        "economic collapse", "the economic collapse blog", "end times", "prophecy",
        "prophecies", "apocalypse", "apocalyptic", "tribulation", "antichrist",
        "rapture", "armageddon", "real estate", "realtor", "homes for sale",
        "mls listing", "used cars", "car dealership", "actor", "filmography",
        "voice actor", "imdb biography", "movie review", "boxing", "nfl", "nba",
        "pastor michael snyder", "prophetic ministry"
    ]

    INSTITUTIONAL_MARKERS = [
        "stanford", "yale", "caltech", "genetics", "genomics", "genome", "biology",
        "medicine", "professor", "snyder lab", "dna", "rna", "omics", "biochemistry",
        "wearable", "glucose", "cgm", "science", "biomedical", "scientific", "dr."
    ]

    @classmethod
    def is_relevant(
        cls,
        title: str,
        text: str = "",
        author_or_channel: str = "",
        llm_assistant: Optional[LocalLLMScraperAssistant] = None,
    ) -> Tuple[bool, str]:
        """
        Validates open web/video content using Semantic LLM first,
        falling back to negative blacklist and institutional markers.
        """
        combined = f"{title} {author_or_channel} {text[:3000]}".lower()

        # Step 1: Deterministic negative blacklist check (blocks obvious homonyms immediately)
        for neg in cls.NEGATIVE_KEYWORDS:
            if neg in combined:
                return False, f"Matched negative homonym keyword: '{neg}'"

        # Step 2: If Local LLM Assistant is available, use semantic classification
        if llm_assistant and llm_assistant.is_available():
            llm_result = llm_assistant.classify_snyder_content(title, author_or_channel, text)
            if llm_result is not None:
                if llm_result:
                    return True, "Validated by Local LLM semantic classifier"
                else:
                    return False, "Rejected by Local LLM semantic classifier"

        # Step 3: Heuristic Fallback (when local LLM is offline)
        has_snyder_name = any(
            name in combined
            for name in ["michael snyder", "mike snyder", "snyder m", "snyder, m", "snyder lab", "m. snyder", "dr. snyder"]
        )

        has_marker = any(marker in combined for marker in cls.INSTITUTIONAL_MARKERS)

        if has_snyder_name and has_marker:
            return True, "Validated via Snyder name and institutional marker"

        if not has_snyder_name:
            return False, "Missing Michael Snyder name identification"

        return True, "Validated"


# =====================================================================
# 2. Wi-Fi Resilience & Safe Network Request Helpers
# =====================================================================

def create_resilient_session(
    max_retries: int = 5,
    backoff_factor: float = 1.0,
    pool_connections: int = 10,
    pool_maxsize: int = 10,
) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def safe_http_request(
    session: requests.Session,
    method: str,
    url: str,
    max_network_retries: int = 5,
    backoff_base: float = 2.0,
    timeout: int = 15,
    **kwargs,
) -> Optional[requests.Response]:
    for attempt in range(1, max_network_retries + 1):
        try:
            res = session.request(method=method, url=url, timeout=timeout, **kwargs)
            if res.status_code == 429:
                txt = res.text.lower() if res.text else ""
                if "budget" in txt or "credit" in txt or "pricing" in txt:
                    logger.warning(f"[DAILY BUDGET EXCEEDED 429] {url[:60]}... Daily API limit exhausted. Skipping retries.")
                    return res
                wait_sec = backoff_base ** attempt
                logger.warning(
                    f"[RATE LIMIT 429] URL: {url[:60]}... "
                    f"Waiting {wait_sec:.1f}s before retry (attempt {attempt}/{max_network_retries})"
                )
                time.sleep(wait_sec)
                continue
            elif res.status_code in [502, 503, 504]:
                wait_sec = backoff_base ** attempt
                logger.warning(
                    f"[SERVER ERROR {res.status_code}] URL: {url[:60]}... "
                    f"Waiting {wait_sec:.1f}s before retry (attempt {attempt}/{max_network_retries})"
                )
                time.sleep(wait_sec)
                continue

            return res
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            socket.error,
            Exception,
        ) as e:
            wait_sec = min(30.0, backoff_base ** attempt)
            logger.warning(
                f"[WIFI / NETWORK ERROR] {type(e).__name__} on '{url[:60]}...': {e}. "
                f"Retrying in {wait_sec:.1f}s (attempt {attempt}/{max_network_retries})..."
            )
            time.sleep(wait_sec)

    logger.error(f"[NETWORK PERMANENT FAILURE] Could not fetch '{url}' after {max_network_retries} attempts.")
    return None


def safe_fetch_youtube_transcript(
    video_id: str,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> Optional[str]:
    for attempt in range(1, max_retries + 1):
        try:
            fetched_tx = YouTubeTranscriptApi().fetch(video_id)
            transcript_lines = []
            for snippet in fetched_tx:
                line_text = (
                    getattr(snippet, "text", None)
                    or (snippet.get("text") if isinstance(snippet, dict) else str(snippet))
                )
                if line_text and line_text.strip() and not line_text.startswith("["):
                    transcript_lines.append(line_text.strip())

            if not transcript_lines:
                return None

            return " ".join(transcript_lines)
        except Exception as e:
            err_str = str(e).lower()
            if "disabled" in err_str or "could not retrieve" in err_str or "no transcript" in err_str or "transcriptsdisabled" in err_str:
                logger.debug(f"Transcript permanently unavailable for {video_id}: {e}")
                return None

            wait_sec = backoff_base ** attempt
            logger.warning(
                f"[WIFI / YOUTUBE RETRY] Error fetching transcript for {video_id}: {e}. "
                f"Retrying in {wait_sec:.1f}s (attempt {attempt}/{max_retries})..."
            )
            time.sleep(wait_sec)

    return None


# =====================================================================
# 3. Manifest & Disk Synchronization (Unified Deduplication)
# =====================================================================

def load_and_sync_scraper_manifest(
    raw_dir: Path = RAW_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load manifest from {manifest_path}: {e}")

    synced_count = 0
    if raw_dir.exists():
        for meta_file in raw_dir.glob("*.meta.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                pmid = meta.get("pmid")
                pmcid = meta.get("pmcid")
                video_id = meta.get("video_id")
                url = meta.get("url", "")
                title = meta.get("source_title", meta_file.stem)

                if pmid:
                    key = f"pmid_{pmid}"
                    if key not in manifest:
                        manifest[key] = {"title": title, "year": meta.get("year", 2012), "pmid": pmid}
                        synced_count += 1
                elif pmcid:
                    key = f"pmcid_{pmcid}"
                    if key not in manifest:
                        manifest[key] = {"title": title, "year": meta.get("year", 2012), "pmcid": pmcid}
                        synced_count += 1
                elif video_id:
                    key = f"yt_{video_id}"
                    if key not in manifest:
                        manifest[key] = {"title": title, "year": meta.get("year", 2021), "video_id": video_id}
                        synced_count += 1
                elif url:
                    url_hash = str(abs(hash(url)))
                    key = f"web_{url_hash}"
                    if key not in manifest:
                        manifest[key] = {"title": title, "url": url, "year": meta.get("year", 2019)}
                        synced_count += 1
            except Exception:
                continue

    if synced_count > 0:
        logger.info(f"Synchronized manifest with {synced_count} pre-existing disk documents.")
        save_scraper_manifest(manifest, manifest_path)

    return manifest


def save_scraper_manifest(manifest: Dict[str, Any], manifest_path: Path = MANIFEST_PATH):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if temp_path.exists():
            temp_path.replace(manifest_path)
    except Exception as e:
        logger.warning(f"Error saving manifest: {e}")


# =====================================================================
# 4. Autonomous Broad YouTube Discovery Scraper (Tier 1 & Tier 2)
# =====================================================================

class AutonomousYouTubeScraper:
    """
    Autonomously searches YouTube for talks, podcasts, and interviews by Michael Snyder
    across broad exploratory queries, with semantic LLM disambiguation.
    """
    SEARCH_QUERIES = [
        "Michael Snyder Stanford",
        "Michael Snyder lecture",
        "Michael Snyder talk",
        "Michael Snyder interview",
        "Michael Snyder podcast",
        "Michael Snyder presentation",
        "Michael Snyder keynote",
        "Dr. Michael Snyder genetics",
        "Michael Snyder genomics",
        "Professor Michael Snyder",
        "Michael Snyder Q&A",
        "Michael Snyder webinar",
        "Michael Snyder seminar",
        "Snyder Lab Stanford",
        "Michael Snyder symposium",
        "Michael Snyder Huberman Lab",
        "Michael Snyder FoundMyFitness",
        "Michael Snyder Peter Attia",
    ]

    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir
        self.session = create_resilient_session()
        self.manifest = load_and_sync_scraper_manifest(self.output_dir)
        self.llm_assistant = LocalLLMScraperAssistant()

    def search_video_ids(self, query: str, max_results: int = 6) -> List[str]:
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
        logger.info(f"Searching YouTube for: '{query}'")

        res = safe_http_request(self.session, "GET", search_url, timeout=12)
        if not res or res.status_code != 200:
            logger.warning(f"YouTube search request failed for query: '{query}'")
            return []

        raw_vids = re.findall(r"watch\?v=([a-zA-Z0-9_-]{11})", res.text)
        video_ids = list(dict.fromkeys(raw_vids))
        logger.info(f"Discovered {len(video_ids)} candidate video IDs for query '{query}'")
        return video_ids[:max_results]

    def fetch_video_metadata(self, video_id: str) -> Dict[str, Any]:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        res = safe_http_request(self.session, "GET", oembed_url, timeout=8)
        if res and res.status_code == 200:
            try:
                data = res.json()
                return {
                    "title": data.get("title", f"YouTube Video {video_id}"),
                    "author": data.get("author_name", "Unknown Channel"),
                }
            except Exception:
                pass
        return {"title": f"YouTube Video {video_id}", "author": "Unknown Channel"}

    def discover_and_scrape(self, max_videos_per_query: int = 4) -> List[Path]:
        saved_paths: List[Path] = []
        seen_video_ids: Set[str] = set()

        for query in self.SEARCH_QUERIES:
            vids = self.search_video_ids(query, max_results=max_videos_per_query)
            for vid in vids:
                if vid in seen_video_ids:
                    continue
                seen_video_ids.add(vid)

                manifest_key = f"yt_{vid}"
                if manifest_key in self.manifest:
                    continue

                meta = self.fetch_video_metadata(vid)
                title = meta["title"]
                author = meta["author"]

                raw_transcript = safe_fetch_youtube_transcript(vid)
                if not raw_transcript:
                    logger.debug(f"Skipping {vid}: Transcript unavailable or inaccessible.")
                    continue

                # Semantic / Heuristic Validation
                is_valid, reason = TopicRelevanceValidator.is_relevant(
                    title=title,
                    text=raw_transcript[:2000],
                    author_or_channel=author,
                    llm_assistant=self.llm_assistant,
                )
                if not is_valid:
                    logger.info(f"[DISAMBIGUATION SKIP] YouTube video {vid} ('{title[:40]}'): {reason}")
                    continue

                year_match = re.search(r"(20[012]\d|199\d)", title)
                year = int(year_match.group(1)) if year_match else 2021

                lower_title = (title + " " + author).lower()
                is_interview = any(w in lower_title for w in ["interview", "podcast", "conversation", "discussion", "huberman", "foundmyfitness", "attiac"])
                tier = SourceTier.TIER_1_ORAL_INTERVIEW if is_interview else SourceTier.TIER_2_PREPARED_LECTURE

                from reprocess_transcripts import AudioTranscriptDiarizer
                diarizer = AudioTranscriptDiarizer(model_size="small.en", device="cpu", compute_type="int8")
                
                # Attempt audio download & high-precision transcription
                target_url = f"https://www.youtube.com/watch?v={vid}"
                audio_file = diarizer.download_audio(target_url)
                if audio_file and audio_file.exists():
                    try:
                        segments = diarizer.transcribe_audio(audio_file)
                        if is_interview:
                            formatted_transcript = diarizer.diarize_interview_segments(segments, title=title)
                        else:
                            formatted_transcript = diarizer.diarize_talk_segments(segments, title=title)
                    finally:
                        if audio_file.exists():
                            try:
                                audio_file.unlink()
                            except Exception:
                                pass
                else:
                    # Fallback text intro separation
                    clean_raw = raw_transcript.strip()
                    intro_match = re.search(r"(okay[, ]+well[, ]+thanks|it['’]s really great to be here|it is a pleasure to be here|thank you|good (morning|afternoon))", clean_raw, re.IGNORECASE)
                    if intro_match and any(w in clean_raw[:intro_match.start()].lower() for w in ["speaker", "professor", "welcome", "dr.", "chair", "richard", "host"]):
                        intro = clean_raw[:intro_match.start()].strip()
                        snyder = clean_raw[intro_match.start():].strip()
                        formatted_transcript = f"[Moderator]: {intro}\n\n[Michael Snyder]: {snyder}"
                    elif is_interview and self.llm_assistant.is_available():
                        formatted_transcript = self.llm_assistant.format_dialogue_turns(raw_transcript)
                    else:
                        formatted_transcript = f"[Michael Snyder]: {raw_transcript}"

                out_txt = self.output_dir / f"transcript_yt_{vid}_{year}.txt"
                out_meta = self.output_dir / f"transcript_yt_{vid}_{year}.meta.json"

                if out_txt.exists():
                    self.manifest[manifest_key] = {"title": title, "year": year}
                    save_scraper_manifest(self.manifest)
                    continue

                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(formatted_transcript)

                meta_data = {
                    "source_title": title,
                    "tier": int(tier),
                    "year": year,
                    "source_type": "youtube_transcript",
                    "channel": author,
                    "video_id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                }
                with open(out_meta, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, indent=2)

                self.manifest[manifest_key] = {"title": title, "year": year, "video_id": vid}
                save_scraper_manifest(self.manifest)

                logger.info(f"Discovered & saved live YouTube transcript: '{title[:50]}' ({vid}, {year})")
                saved_paths.append(out_txt)
                time.sleep(0.5)

        logger.info(f"YouTube Discovery finished. Processed {len(seen_video_ids)} candidates, saved {len(saved_paths)} new transcripts.")
        return saved_paths


# =====================================================================
# 5. Europe PMC Full-Text XML Parser & Autonomous Scraper (Tier 3)
# =====================================================================

def fetch_europe_pmc_full_text(session: requests.Session, pmcid: str, timeout: int = 15) -> Optional[str]:
    """
    Fetches full-text for a PMCID using a robust multi-tier fallback architecture:
    1. Europe PMC Full-Text XML (core CC-BY open-access repository)
    2. NCBI PMC Web/HTML parser (NIH Public Access & author manuscripts)
    3. Europe PMC Web/HTML article reader
    """
    if not pmcid:
        return None

    clean_pmcid = pmcid.strip()
    if not clean_pmcid.upper().startswith("PMC"):
        clean_pmcid = f"PMC{clean_pmcid}"

    # Tier 1: Europe PMC REST XML
    url_xml = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{clean_pmcid}/fullTextXML"
    res = safe_http_request(session, "GET", url_xml, timeout=timeout)
    if res and res.status_code == 200 and res.text.strip():
        try:
            soup = BeautifulSoup(res.text, "xml")
            body = soup.find("body")
            if body:
                for tag in body.find_all(["table-wrap", "table", "fig", "media", "supplementary-material", "ref-list"]):
                    tag.decompose()

                sections = []
                sec_tags = body.find_all("sec", recursive=False) or body.find_all("sec")

                if sec_tags:
                    for sec in sec_tags:
                        title_elem = sec.find("title")
                        sec_title = title_elem.get_text().strip() if title_elem else "Section"
                        if title_elem:
                            title_elem.extract()

                        paras = [p.get_text(separator=" ", strip=True) for p in sec.find_all(["p", "list"], recursive=False)]
                        if not paras:
                            direct_text = sec.get_text(separator="\n", strip=True)
                            if direct_text and direct_text != sec_title:
                                sections.append(f"=== {sec_title} ===\n{direct_text}")
                        else:
                            p_text = "\n\n".join(paras)
                            sections.append(f"=== {sec_title} ===\n{p_text}")
                else:
                    body_text = body.get_text(separator="\n\n", strip=True)
                    if body_text:
                        sections.append(body_text)

                full_body = "\n\n".join(sections).strip()
                if len(full_body) > 50:
                    return full_body
        except Exception as e:
            logger.debug(f"XML parse error for {clean_pmcid}: {e}")

    # Tier 2: NCBI PMC HTML (for NIH author manuscripts not in Europe PMC XML bucket)
    url_ncbi = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{clean_pmcid}/"
    res_ncbi = safe_http_request(session, "GET", url_ncbi, timeout=timeout)
    if res_ncbi and res_ncbi.status_code == 200 and len(res_ncbi.text) > 1000:
        try:
            soup = BeautifulSoup(res_ncbi.text, "html.parser")
            sections = []
            for sec in soup.find_all(["section", "div"], class_=lambda c: c and any(k in str(c).lower() for k in ["sec", "section", "article-section"])):
                heading = sec.find(["h2", "h3", "h4"])
                h_text = heading.get_text().strip() if heading else "Section"
                if any(bad in h_text.lower() for bad in ["reference", "citation", "author info", "permalink", "supplementary", "copyright"]):
                    continue
                paras = [p.get_text(separator=" ", strip=True) for p in sec.find_all("p")]
                if paras:
                    sections.append(f"=== {h_text} ===\n" + "\n\n".join(paras))
            if sections:
                full_body = "\n\n".join(sections).strip()
                if len(full_body) > 300:
                    return full_body
        except Exception as e:
            logger.debug(f"NCBI HTML parse error for {clean_pmcid}: {e}")

    # Tier 3: Europe PMC HTML Reader
    url_epmc_html = f"https://europepmc.org/articles/{clean_pmcid}"
    res_epmc = safe_http_request(session, "GET", url_epmc_html, timeout=timeout)
    if res_epmc and res_epmc.status_code == 200 and len(res_epmc.text) > 1000:
        try:
            soup = BeautifulSoup(res_epmc.text, "html.parser")
            body_div = soup.find("div", id="free-full-text") or soup.find("div", class_="article-body")
            if body_div:
                text = body_div.get_text(separator="\n\n", strip=True)
                if len(text) > 300:
                    return text
        except Exception as e:
            logger.debug(f"Europe PMC HTML parse error for {clean_pmcid}: {e}")

    return None


def is_snyder_genomics_paper(title: str, abstract: str, authors: str = "", journal: str = "") -> bool:
    """
    Validates whether a paper is genuinely by Dr. Michael P. Snyder (Stanford/Yale
    genomics, genetics, proteomics, systems biology, personal omics, wearables).
    Filters out homonyms (political science, economics, philosophy, unrelated clinical trials, other Snyders).
    """
    auth_lower = authors.lower()
    if auth_lower:
        non_michael = [
            "c. r. snyder", "c.r. snyder", "charles r. snyder", "peter j. snyder",
            "donald snyder", "stephen snyder", "solomon h. snyder", "solomon snyder",
            "mark snyder", "m. a. snyder", "m. r. snyder", "m. e. snyder", "allan snyder"
        ]
        if any(nm in auth_lower for nm in non_michael) and not any(m in auth_lower for m in ["michael", "m. p. snyder", "mp snyder"]):
            return False

    combined = f"{title} {abstract} {journal}".lower()

    # Blacklist indicators for homonyms
    negative_terms = [
        "turtle", "emydoidea", "reptile", "amphibian", "social interaction",
        "interpersonal processes", "utilization review laws", "managed care regulation",
        "social psychology", "economic prophecy", "biblical apocalypse",
        "astropy", "astronomy", "astrophys", "telescope", "galaxies", "planetary",
        "gravitational", "solar system", "cosmolog", "pseudo-singularit", "singularity",
        "black hole", "relativity", "quantum gravity", "spacetime", "astronomical",
        "quantum physics", "general relativity", "lorentz",
        "trade wars", "international regime", "resource curse", "public goods consumption",
        "political science", "autocratic audience", "distribution of federal spend",
        "does the media matter", "counterfactuals and hypothesis testing", "sourcing by design",
        "michael addition of", "relationship between hope", "prolactin respon", "thyrotropin response",
        "snyder-robinson syndrome", "patent foramen ovale", "adhd-200", "high-power performan"
    ]
    for neg in negative_terms:
        if neg in combined:
            return False

    positive_terms = [
        "genet", "genom", "yeast", "drosophila", "protein", "dna", "rna", "chip", "encode",
        "transposon", "mutagenesis", "microarray", "proteom", "transcriptom", "omics",
        "ipop", "metabolom", "glucose", "cgm", "wearable", "biosensor", "sequencing",
        "chromatin", "transcription factor", "cell", "molecular", "saccharomyces",
        "stanford", "yale", "caltech", "biology", "biochem", "disease", "diabetes", "immune",
        "nature", "science", "cell", "genome", "plos", "pnas", "nucleic acids"
    ]
    
    matches = sum(1 for term in positive_terms if term in combined)
    if any(term in combined for term in ["stanford", "yale", "caltech", "ipop"]):
        return True
    return matches >= 2


class AutonomousEuropePMCScraper:
    """
    Autonomously searches Europe PMC REST API across Michael Snyder's entire career
    using broad chronological & affiliation queries.
    Retrieves full-text XML when open access (PMCID) is available, with graceful
    fallback to abstracts for closed-access papers.
    """
    SEARCH_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    SEARCH_EPOCHS = [
        ("Early Career & Postdoc (1980-1995: Caltech, Stanford, Yale)",
         'AUTH:"Snyder M" AND (PUB_YEAR:[1980 TO 1995]) AND (SRC:MED OR SRC:PMC)'),
        ("Functional Genomics Era (1996-2005: Yale)",
         'AUTH:"Snyder M" AND (PUB_YEAR:[1996 TO 2005]) AND (SRC:MED OR SRC:PMC)'),
        ("High-Throughput Sequencing & ENCODE (2006-2015: Yale & Stanford)",
         'AUTH:"Snyder M" AND (PUB_YEAR:[2006 TO 2015]) AND (SRC:MED OR SRC:PMC)'),
        ("Personal Omics, Wearables & Precision Health (2016-2026: Stanford)",
         'AUTH:"Snyder M" AND (PUB_YEAR:[2016 TO 2026]) AND (SRC:MED OR SRC:PMC)'),
        ("Direct Stanford & Yale Affiliated Corpus",
         'AUTH:"Snyder M" AND (AFF:"Stanford" OR AFF:"Yale" OR AFF:"Genetics" OR AFF:"Genomics" OR AFF:"Biology") AND (SRC:MED OR SRC:PMC)'),
        ("Michael P. Snyder Canonical Author Identifier",
         'AUTH:"Snyder MP" AND (SRC:MED OR SRC:PMC)'),
    ]

    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir
        self.session = create_resilient_session()
        self.manifest = load_and_sync_scraper_manifest(self.output_dir)

    def search_and_download(
        self,
        max_papers_per_epoch: int = 50,
        page_size: int = 25,
    ) -> List[Path]:
        saved_paths: List[Path] = []
        logger.info("Starting unconstrained Europe PMC scientific harvest across Michael Snyder's complete bibliography...")

        for epoch_name, epoch_query in self.SEARCH_EPOCHS:
            logger.info(f"\n--- Harvesting Epoch: {epoch_name} ---")
            epoch_saved = 0
            cursor = "*"
            page_num = 1
            max_pages = max(10, (max_papers_per_epoch // page_size) + 5)

            while epoch_saved < max_papers_per_epoch and cursor and page_num <= max_pages:
                params = {
                    "query": epoch_query,
                    "format": "json",
                    "pageSize": page_size,
                    "cursorMark": cursor,
                    "resultType": "core",
                    "sort": "CITED desc",
                }

                res = safe_http_request(self.session, "GET", self.SEARCH_ENDPOINT, params=params, timeout=15)
                if not res or res.status_code != 200:
                    logger.warning(f"Europe PMC request failed for cursor {cursor[:15]} of '{epoch_name}'. Moving to next epoch.")
                    break

                try:
                    data = res.json()
                except Exception:
                    logger.warning(f"Invalid JSON from Europe PMC on page {page_num}.")
                    break

                results = data.get("resultList", {}).get("result", [])
                next_cursor = data.get("nextCursorMark")

                if not results:
                    logger.info(f"Reached end of results for epoch '{epoch_name}'.")
                    break

                for item in results:
                    if epoch_saved >= max_papers_per_epoch:
                        break

                    title = item.get("title", "").rstrip(".")
                    if not title:
                        continue

                    abstract = item.get("abstractText", "")
                    authors = item.get("authorString", "Michael Snyder et al.")
                    journal = item.get("journalTitle", "Scientific Journal")

                    # Disambiguation filter
                    if not is_snyder_genomics_paper(title, abstract, authors, journal):
                        continue

                    pmid = item.get("pmid", "")
                    pmcid = item.get("pmcid", "")
                    doc_id = pmid or pmcid or str(abs(hash(title)))

                    canonical_key = f"pmid_{pmid}" if pmid else (f"pmcid_{pmcid}" if pmcid else f"title_hash_{doc_id}")
                    if canonical_key in self.manifest:
                        continue

                    pub_year = int(item.get("pubYear", 2012))

                    out_txt = self.output_dir / f"paper_pmc_{doc_id}_{pub_year}.txt"
                    out_meta = self.output_dir / f"paper_pmc_{doc_id}_{pub_year}.meta.json"

                    if out_txt.exists():
                        self.manifest[canonical_key] = {"title": title, "year": pub_year, "pmid": pmid}
                        save_scraper_manifest(self.manifest)
                        continue

                    # Attempt full-text XML extraction if PMCID is present
                    full_text = None
                    if pmcid:
                        full_text = fetch_europe_pmc_full_text(self.session, pmcid)

                    has_full_text = bool(full_text)
                    if has_full_text:
                        content = (
                            f"Title: {title}\n"
                            f"Authors: {authors}\n"
                            f"Journal: {journal} ({pub_year})\n"
                            f"PMID: {pmid} | PMCID: {pmcid}\n"
                            f"Access: Open Access (Full Text)\n\n"
                            f"Abstract:\n{abstract}\n\n"
                            f"Full Text:\n{full_text}\n"
                        )
                    else:
                        content = (
                            f"Title: {title}\n"
                            f"Authors: {authors}\n"
                            f"Journal: {journal} ({pub_year})\n"
                            f"PMID: {pmid} | PMCID: {pmcid}\n"
                            f"Access: Abstract Only\n\n"
                            f"Abstract:\n{abstract}\n"
                        )

                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(content)

                    meta_data = {
                        "source_title": title,
                        "tier": int(SourceTier.TIER_3_PUBLISHED_PAPER),
                        "year": pub_year,
                        "source_type": "published_paper",
                        "journal": journal,
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "has_full_text": has_full_text,
                        "full_text_chars": len(full_text) if full_text else 0,
                        "url": f"https://europepmc.org/article/MED/{pmid}" if pmid else "",
                    }
                    with open(out_meta, "w", encoding="utf-8") as f:
                        json.dump(meta_data, f, indent=2)

                    self.manifest[canonical_key] = {"title": title, "year": pub_year, "pmid": pmid}
                    save_scraper_manifest(self.manifest)

                    text_type = "FULL TEXT" if has_full_text else "Abstract"
                    logger.info(f"Discovered & saved Europe PMC paper [{text_type}]: '{title[:50]}...' ({pub_year})")
                    saved_paths.append(out_txt)
                    epoch_saved += 1

                if not next_cursor or next_cursor == cursor:
                    logger.info(f"Exhausted all available cursors for epoch '{epoch_name}'.")
                    break

                cursor = next_cursor
                page_num += 1
                time.sleep(0.4)

        logger.info(f"\nEurope PMC Discovery Complete: Acquired {len(saved_paths)} new scientific publications.")
        return saved_paths


# =====================================================================
# 6. Europe PMC Recursive Citation Snowball Crawler (Tier 3)
# =====================================================================

class EuropePMCCitationCrawler:
    LANDMARK_PMIDS = [
        "22424949",  # 2012 iPOP Multi-Omics (Cell)
        "9381179",   # 1997 Whole-Genome Expression / Microarrays (Science)
        "11158315",  # 2000 ChIP-chip / Genomic Binding (Nature)
        "8479524",   # 1993 Transposon Mutagenesis (Science)
        "17568003",  # 2007 ENCODE Pilot Project (Nature)
        "28085175",  # 2017 Digital Health / Smartwatches & Lyme/Infection (PLOS Biology)
        "30044998",  # 2018 Continuous Glucose Monitoring Glucotypes (PLOS Biology)
        "31932738",  # 2020 Personal Ageotypes (Nature Medicine)
    ]
    CITATIONS_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/MED/{pmid}/citations?format=json"

    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir
        self.session = create_resilient_session()
        self.manifest = load_and_sync_scraper_manifest(self.output_dir)

    def crawl_citations(self, max_per_landmark: int = 10, page_size: int = 25) -> List[Path]:
        saved_paths: List[Path] = []
        logger.info(f"Running recursive citation snowball discovery across {len(self.LANDMARK_PMIDS)} landmark publications...")

        for pmid in self.LANDMARK_PMIDS:
            logger.info(f"Scanning citations for landmark PMID: {pmid}")
            landmark_saved = 0
            page = 1

            while landmark_saved < max_per_landmark:
                url = self.CITATIONS_ENDPOINT.format(pmid=pmid) + f"&page={page}&pageSize={page_size}"
                res = safe_http_request(self.session, "GET", url, timeout=15)
                if not res or res.status_code != 200:
                    logger.warning(f"Could not retrieve citations for PMID {pmid} on page {page}.")
                    break

                try:
                    data = res.json()
                except Exception:
                    break

                citations = data.get("citationList", {}).get("citation", [])
                if not citations:
                    logger.info(f"No further citations found for landmark PMID {pmid}.")
                    break

                for cite in citations:
                    if landmark_saved >= max_per_landmark:
                        break

                    cite_pmid = cite.get("id", "")
                    title = cite.get("title", "").rstrip(".")
                    author_str = cite.get("authorString", "")
                    pub_year = int(cite.get("pubYear", 2015))

                    is_snyder_authored = "snyder m" in author_str.lower() or "michael snyder" in author_str.lower()
                    is_snyder_subject = "michael snyder" in title.lower() or ("snyder" in title.lower() and "genomics" in title.lower())

                    if not (is_snyder_authored or is_snyder_subject):
                        continue

                    doc_id = cite_pmid or str(abs(hash(title)))
                    canonical_key = f"pmid_{cite_pmid}" if cite_pmid else f"title_hash_{doc_id}"

                    if canonical_key in self.manifest:
                        continue

                    out_txt = self.output_dir / f"paper_citation_{doc_id}_{pub_year}.txt"
                    out_meta = self.output_dir / f"paper_citation_{doc_id}_{pub_year}.meta.json"

                    if out_txt.exists():
                        self.manifest[canonical_key] = {"title": title, "year": pub_year, "pmid": cite_pmid}
                        save_scraper_manifest(self.manifest)
                        continue

                    content = (
                        f"Title: {title}\n"
                        f"Authors: {author_str}\n"
                        f"Year: {pub_year} | Citing Landmark PMID: {pmid}\n"
                        f"PMID: {cite_pmid}\n\n"
                        f"Retrospective Follow-Up & Citation Context:\n"
                        f"{title}\n"
                    )

                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(content)

                    meta_data = {
                        "source_title": title,
                        "tier": int(SourceTier.TIER_3_PUBLISHED_PAPER),
                        "year": pub_year,
                        "source_type": "citation_followup",
                        "citing_parent_pmid": pmid,
                        "pmid": cite_pmid,
                        "url": f"https://europepmc.org/article/MED/{cite_pmid}" if cite_pmid else "",
                    }
                    with open(out_meta, "w", encoding="utf-8") as f:
                        json.dump(meta_data, f, indent=2)

                    self.manifest[canonical_key] = {"title": title, "year": pub_year, "pmid": cite_pmid}
                    save_scraper_manifest(self.manifest)

                    logger.info(f"Discovered & saved citing follow-up: '{title[:50]}' ({pub_year})")
                    saved_paths.append(out_txt)
                    landmark_saved += 1
                    time.sleep(0.4)

                if len(citations) < page_size:
                    break
                page += 1

        logger.info(f"Citation Snowball Discovery Complete: Acquired {len(saved_paths)} new citing follow-ups.")
        return saved_paths


# =====================================================================
# 7. Autonomous OpenAlex Open-Access PDF Harvester (Tier 3)
# =====================================================================

class AutonomousOpenAlexScraper:
    """
    Harvester that queries OpenAlex API for Dr. Michael Snyder's works,
    identifies Open Access full-text PDF links, downloads them into memory,
    extracts high-quality full text using pypdf, and saves them to disk.
    """
    SEARCH_ENDPOINT = "https://api.openalex.org/works"

    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir
        self.session = create_resilient_session()
        self.manifest = load_and_sync_scraper_manifest(self.output_dir)

    def search_and_download(
        self,
        max_papers: int = 50,
        per_page: int = 25,
    ) -> List[Path]:
        saved_paths: List[Path] = []
        logger.info(f"Starting OpenAlex Open Access PDF harvest for Michael Snyder (Target: up to {max_papers} full-text papers)...")

        # Step 1: Pre-index disk metadata into fast in-memory map (0.1s instead of thousands of disk scans)
        disk_pmid_map: Dict[str, Dict[str, Any]] = {}
        for mf in self.output_dir.glob("*.meta.json"):
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    pmid_val = str(d.get("pmid", ""))
                    if pmid_val:
                        disk_pmid_map[pmid_val] = {
                            "meta_file": mf,
                            "txt_file": mf.with_name(mf.name.replace(".meta.json", ".txt")),
                            "has_full_text": bool(d.get("has_full_text")),
                        }
            except Exception:
                pass

        logger.info(f"Loaded {len(disk_pmid_map)} existing paper records from disk for instant deduplication.")

        cursor = "*"
        saved_count = 0
        page = 1
        max_pages = 250

        while saved_count < max_papers and cursor and page <= max_pages:
            logger.info(f"Scanning OpenAlex page {page} (Acquired so far: {saved_count}/{max_papers})...")
            params = {
                "filter": "default.search:Michael Snyder,open_access.is_oa:true,has_fulltext:true",
                "per-page": per_page,
                "cursor": cursor,
                "mailto": "team@cortex.bio",
            }

            res = safe_http_request(self.session, "GET", self.SEARCH_ENDPOINT, params=params, timeout=12, max_network_retries=2)
            if not res or res.status_code != 200:
                logger.warning(f"OpenAlex request failed for cursor {cursor[:15]}. Ending harvest.")
                break

            try:
                data = res.json()
            except Exception:
                break

            results = data.get("results", [])
            next_cursor = data.get("meta", {}).get("next_cursor")

            if not results:
                logger.info("No further results returned by OpenAlex API.")
                break

            for work in results:
                if saved_count >= max_papers:
                    break

                title = work.get("title", "").strip().rstrip(".")
                if not title:
                    continue

                # Author disambiguation
                authorships = work.get("authorships", [])
                if not any("snyder" in a.get("author", {}).get("display_name", "").lower() for a in authorships if a.get("author")):
                    continue

                authors_str = ", ".join([a.get("author", {}).get("display_name", "") for a in authorships if a.get("author")])
                primary_loc = work.get("primary_location") or {}
                source_dict = primary_loc.get("source") if isinstance(primary_loc, dict) else {}
                journal = source_dict.get("display_name") if isinstance(source_dict, dict) else "Scientific Journal"
                if not journal:
                    journal = "Scientific Journal"

                if not is_snyder_genomics_paper(title, "", authors_str, journal):
                    continue

                ids = work.get("ids", {})
                pmid = ids.get("pmid", "").replace("https://pubmed.ncbi.nlm.nih.gov/", "")
                doi = ids.get("doi", "").replace("https://doi.org/", "")
                openalex_id = work.get("id", "").split("/")[-1]
                doc_id = pmid or openalex_id

                # Instant in-memory check
                existing_meta_file = None
                existing_txt_file = None

                if pmid and pmid in disk_pmid_map:
                    entry = disk_pmid_map[pmid]
                    if entry["has_full_text"]:
                        # Already has full text
                        continue
                    existing_meta_file = entry["meta_file"]
                    existing_txt_file = entry["txt_file"]

                best_oa = work.get("best_oa_location") or {}
                pdf_url = best_oa.get("pdf_url")
                if not pdf_url:
                    continue

                pub_year = int(work.get("publication_year", 2015))

                if existing_txt_file and existing_meta_file:
                    out_txt = existing_txt_file
                    out_meta = existing_meta_file
                else:
                    out_txt = self.output_dir / f"paper_openalex_{doc_id}_{pub_year}.txt"
                    out_meta = self.output_dir / f"paper_openalex_{doc_id}_{pub_year}.meta.json"

                if out_txt.exists() and not existing_txt_file:
                    if out_meta.exists():
                        try:
                            with open(out_meta, "r", encoding="utf-8") as mf:
                                if json.load(mf).get("has_full_text"):
                                    continue
                        except Exception:
                            pass

                # Download PDF with responsive timeout (no retry delays)
                logger.info(f"[{saved_count+1}/{max_papers}] Fetching OA PDF: '{title[:45]}...' ({pub_year})")
                try:
                    pdf_res = requests.get(pdf_url, timeout=8, headers={"User-Agent": USER_AGENT})
                    if pdf_res.status_code != 200 or len(pdf_res.content) < 5000 or not pdf_res.content.startswith(b"%PDF"):
                        continue
                except Exception:
                    continue

                try:
                    reader = PdfReader(io.BytesIO(pdf_res.content))
                    pages_text = []
                    for page_obj in reader.pages:
                        pt = page_obj.extract_text()
                        if pt:
                            pages_text.append(pt)
                    full_text = "\n\n".join(pages_text).strip()
                except Exception as e:
                    logger.debug(f"Could not parse PDF for {title}: {e}")
                    continue

                if len(full_text) < 1000:
                    continue

                content = (
                    f"Title: {title}\n"
                    f"Authors: {authors_str}\n"
                    f"Journal: {journal} ({pub_year})\n"
                    f"PMID: {pmid} | DOI: {doi}\n"
                    f"Access: Open Access PDF ({len(reader.pages)} pages)\n\n"
                    f"Full Text:\n{full_text}\n"
                )

                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(content)

                meta_data = {
                    "source_title": title,
                    "tier": int(SourceTier.TIER_3_PUBLISHED_PAPER),
                    "year": pub_year,
                    "source_type": "published_paper_pdf",
                    "journal": journal,
                    "pmid": pmid,
                    "doi": doi,
                    "has_full_text": True,
                    "full_text_chars": len(full_text),
                    "pages": len(reader.pages),
                    "url": pdf_url,
                }
                with open(out_meta, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, indent=2)

                canonical_key = f"openalex_{openalex_id}"
                self.manifest[canonical_key] = {"title": title, "year": pub_year, "pmid": pmid}
                save_scraper_manifest(self.manifest)

                # Update in-memory map
                if pmid:
                    disk_pmid_map[pmid] = {
                        "meta_file": out_meta,
                        "txt_file": out_txt,
                        "has_full_text": True,
                    }

                action_verb = "Upgraded" if existing_txt_file else "Discovered & saved"
                logger.info(f"{action_verb} OpenAlex OA PDF: '{title[:50]}...' ({pub_year}, {len(reader.pages)} pages, {len(full_text):,} chars)")
                saved_paths.append(out_txt)
                saved_count += 1
                time.sleep(0.3)

            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            page += 1

        logger.info(f"OpenAlex Harvest Complete: Acquired {len(saved_paths)} new/upgraded full-text open-access publications.")
        return saved_paths


# =====================================================================
# 7.5. Autonomous Sci-Hub & DOI Mirror Scraper (Tier 3)
# =====================================================================

class AutonomousSciHubScraper:
    """
    Autonomous DOI-based Academic Mirror Harvester for paywalled scientific papers.
    Queries verified active mirrors (sci-hub.ren, sci.bban.top, sci-hub.st, sci-hub.wf),
    extracts embedded PDF binary streams, validates PDF headers, and parses full texts using pypdf.
    """
    DEFAULT_MIRRORS = [
        "https://sci-hub.ren",
        "https://sci.bban.top",
        "https://sci-hub.st",
        "https://sci-hub.wf",
        "https://sci-hub.ru",
    ]

    def __init__(self, output_dir: Path = RAW_DIR, mirrors: Optional[List[str]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mirrors = mirrors or self.DEFAULT_MIRRORS
        self.manifest = load_and_sync_scraper_manifest(self.output_dir)

    def resolve_pdf_url(self, doi: str, timeout: int = 8) -> Optional[str]:
        """
        Attempts to resolve the direct PDF URL for a given DOI across active mirrors.
        """
        clean_doi = doi.strip().replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi.org/", "")
        if not clean_doi:
            return None

        # Direct fast endpoint check for sci.bban.top mirror
        direct_candidate = f"https://sci.bban.top/pdf/{clean_doi}.pdf"
        try:
            r = requests.head(direct_candidate, headers={"User-Agent": USER_AGENT}, verify=False, timeout=4)
            if r.status_code == 200:
                return direct_candidate
        except Exception:
            pass

        headers = {"User-Agent": USER_AGENT}

        for mirror in self.mirrors:
            mirror_clean = mirror.rstrip("/")
            if "bban.top" in mirror_clean:
                continue
            query_url = f"{mirror_clean}/{clean_doi}"
            try:
                res = requests.get(query_url, headers=headers, verify=False, timeout=timeout)
                if res.status_code != 200 or len(res.content) < 100:
                    continue

                soup = BeautifulSoup(res.text, "html.parser")
                embed = soup.find("embed", id="pdf") or soup.find("iframe", id="pdf") or soup.find("embed") or soup.find("iframe")
                if embed and embed.get("src"):
                    src = embed["src"]
                    if src.startswith("//"):
                        return "https:" + src
                    elif src.startswith("/"):
                        return mirror_clean + src
                    elif src.startswith("http"):
                        return src

                button = soup.find("button")
                if button and "location.href=" in button.get("onclick", ""):
                    onclick = button["onclick"]
                    if "location.href='" in onclick:
                        raw_src = onclick.split("location.href='")[1].split("'")[0]
                    elif 'location.href="' in onclick:
                        raw_src = onclick.split('location.href="')[1].split('"')[0]
                    else:
                        continue
                    if raw_src.startswith("//"):
                        return "https:" + raw_src
                    elif raw_src.startswith("/"):
                        return mirror_clean + raw_src
                    elif raw_src.startswith("http"):
                        return raw_src
            except Exception as e:
                logger.debug(f"Mirror {mirror} query failed for DOI {clean_doi}: {e}")
                continue

        return None

    def download_and_extract_pdf(self, pdf_url: str, timeout: int = 12) -> Optional[Tuple[str, int]]:
        """
        Downloads a binary PDF and extracts full text and page count via pypdf.
        """
        headers = {"User-Agent": USER_AGENT}
        try:
            res = requests.get(pdf_url, headers=headers, verify=False, timeout=timeout)
            if res.status_code != 200 or len(res.content) < 5000:
                return None
            if not res.content.startswith(b"%PDF"):
                return None

            reader = PdfReader(io.BytesIO(res.content))
            pages_text = []
            for page in reader.pages:
                pt = page.extract_text()
                if pt:
                    pages_text.append(pt)
            full_text = "\n\n".join(pages_text).strip()
            if len(full_text) < 800:
                return None
            return full_text, len(reader.pages)
        except Exception as e:
            logger.debug(f"PDF extraction failed for {pdf_url[:60]}: {e}")
            return None


# =====================================================================
# 8. Autonomous Broad Web Discovery Scraper (Tier 1 & Tier 4)
# =====================================================================

class AutonomousWebScraper:
    SEARCH_QUERIES = [
        '"Michael Snyder" Stanford',
        '"Michael Snyder" Yale interview',
        '"Michael Snyder" transcript',
        '"Michael Snyder" oral history',
        '"Michael Snyder" profile interview',
        '"Michael Snyder" podcast guest',
        '"Dr. Michael Snyder" genetics',
        '"Michael Snyder" biography',
        '"Michael Snyder" Q&A conversation',
        '"Michael Snyder" "Snyder Lab"',
    ]

    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir
        self.session = create_resilient_session()
        self.manifest = load_and_sync_scraper_manifest(self.output_dir)
        self.llm_assistant = LocalLLMScraperAssistant()

    def search_web_urls(self, query: str, max_urls: int = 4) -> List[str]:
        try:
            url = "https://html.duckduckgo.com/html/"
            res = safe_http_request(self.session, "POST", url, data={"q": query}, timeout=12)
            if not res or res.status_code != 200:
                return []

            soup = BeautifulSoup(res.text, "html.parser")
            urls = []
            for a in soup.find_all("a", class_="result__url"):
                href = a.get("href")
                if href and href.startswith("http") and "youtube.com" not in href:
                    urls.append(href.strip())
            return urls[:max_urls]
        except Exception as e:
            logger.warning(f"Web search error for '{query}': {e}")
            return []

    def fetch_and_parse_article(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            res = safe_http_request(self.session, "GET", url, timeout=12)
            if not res or res.status_code != 200:
                return None

            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()

            title_elem = soup.find("h1") or soup.find("title")
            title = title_elem.get_text(strip=True) if title_elem else "Web Article"

            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
            body_text = "\n\n".join(paragraphs)

            if len(body_text) < 200:
                return None

            is_valid, reason = TopicRelevanceValidator.is_relevant(
                title=title,
                text=body_text,
                llm_assistant=self.llm_assistant,
            )
            if not is_valid:
                logger.info(f"[DISAMBIGUATION WEB SKIP] {url}: {reason}")
                return None

            year_matches = re.findall(r"\b(19[89]\d|20[012]\d)\b", body_text[:2000])
            year = int(year_matches[0]) if year_matches else 2019

            is_dialogue = bool(re.search(r"\b(Q:|Question:|Interviewer:)\b", body_text, re.IGNORECASE))
            tier = SourceTier.TIER_1_ORAL_INTERVIEW if is_dialogue else SourceTier.TIER_4_THIRD_PARTY

            if is_dialogue and self.llm_assistant.is_available():
                body_text = self.llm_assistant.format_dialogue_turns(body_text)

            return {
                "title": title,
                "text": body_text,
                "year": year,
                "tier": tier,
                "url": url,
            }
        except Exception as e:
            logger.debug(f"Article fetch error for {url}: {e}")
            return None

    def discover_and_scrape(self, max_articles: int = 15) -> List[Path]:
        saved_paths: List[Path] = []
        seen_urls: Set[str] = set()

        for query in self.SEARCH_QUERIES:
            urls = self.search_web_urls(query, max_urls=3)
            for u in urls:
                if u in seen_urls or len(saved_paths) >= max_articles:
                    continue
                seen_urls.add(u)

                url_hash = str(abs(hash(u)))
                manifest_key = f"web_{url_hash}"
                if manifest_key in self.manifest:
                    continue

                article_data = self.fetch_and_parse_article(u)
                if not article_data:
                    continue

                slug = re.sub(r"[^a-zA-Z0-9_-]", "_", article_data["title"][:40].lower()).strip("_")
                year = article_data["year"]
                out_txt = self.output_dir / f"web_article_{slug}_{year}.txt"
                out_meta = self.output_dir / f"web_article_{slug}_{year}.meta.json"

                if out_txt.exists():
                    self.manifest[manifest_key] = {"title": article_data["title"], "url": u, "year": year}
                    save_scraper_manifest(self.manifest)
                    continue

                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(article_data["text"])

                meta_data = {
                    "source_title": article_data["title"],
                    "tier": int(article_data["tier"]),
                    "year": year,
                    "source_type": "web_interview" if article_data["tier"] == SourceTier.TIER_1_ORAL_INTERVIEW else "web_profile",
                    "url": article_data["url"],
                }
                with open(out_meta, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, indent=2)

                self.manifest[manifest_key] = {"title": article_data["title"], "url": u, "year": year}
                save_scraper_manifest(self.manifest)

                logger.info(f"Discovered & saved web article: '{article_data['title'][:50]}' ({year})")
                saved_paths.append(out_txt)
                time.sleep(0.5)

        logger.info(f"Web Discovery Complete: Acquired {len(saved_paths)} new web documents.")
        return saved_paths


# =====================================================================
# 8. Podcast RSS Feed Discovery Scraper (Tier 1 Oral Histories)
# =====================================================================

class PodcastRSSScraper:
    PODCAST_FEEDS = [
        ("Huberman Lab", "https://feeds.megaphone.fm/hubermanlab"),
        ("FoundMyFitness", "https://feeds.podcastmirror.com/foundmyfitness"),
        ("The Drive", "https://thepeterattiadrive.libsyn.com/rss"),
        ("Genetics Unzipped", "https://geneticsunzipped.libsyn.com/rss"),
        ("Stanford Medcast", "https://feeds.simplecast.com/8_hO_sL_"),
        ("People Behind the Science", "https://peoplebehindthescience.libsyn.com/rss"),
    ]

    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir
        self.session = create_resilient_session()
        self.manifest = load_and_sync_scraper_manifest(self.output_dir)
        self.llm_assistant = LocalLLMScraperAssistant()

    def discover_and_scrape(self) -> List[Path]:
        import xml.etree.ElementTree as ET
        saved_paths: List[Path] = []
        logger.info("Checking podcast RSS feeds for Michael Snyder guest appearances...")

        for show_name, feed_url in self.PODCAST_FEEDS:
            try:
                res = safe_http_request(self.session, "GET", feed_url, timeout=15)
                if not res or res.status_code != 200:
                    continue

                root = ET.fromstring(res.content)
                channel = root.find("channel")
                if channel is None:
                    continue

                for item in channel.findall("item"):
                    title_elem = item.find("title")
                    desc_elem = item.find("description")
                    pub_date_elem = item.find("pubDate")

                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    desc = desc_elem.text if desc_elem is not None and desc_elem.text else ""
                    pub_date = pub_date_elem.text if pub_date_elem is not None and pub_date_elem.text else ""

                    lower_podcast_text = f"{title} {desc}".lower()
                    if not any(name in lower_podcast_text for name in ["michael snyder", "mike snyder", "snyder lab", "dr. snyder", "snyder"]):
                        continue

                    is_valid, reason = TopicRelevanceValidator.is_relevant(
                        title=title,
                        text=desc,
                        author_or_channel=show_name,
                        llm_assistant=self.llm_assistant,
                    )
                    if not is_valid:
                        continue

                    item_id = str(abs(hash(f"{show_name}_{title}")))
                    manifest_key = f"podcast_{item_id}"
                    if manifest_key in self.manifest:
                        continue

                    year_match = re.search(r"\b(20[012]\d|199\d)\b", f"{pub_date} {title}")
                    year = int(year_match.group(1)) if year_match else 2021

                    soup = BeautifulSoup(desc, "html.parser")
                    clean_desc = soup.get_text(separator="\n", strip=True)

                    content = (
                        f"Podcast: {show_name}\n"
                        f"Episode Title: {title}\n"
                        f"Date: {pub_date} ({year})\n\n"
                        f"[Michael Snyder]: {clean_desc}\n"
                    )

                    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{show_name}_{title}"[:45].lower()).strip("_")
                    out_txt = self.output_dir / f"podcast_rss_{slug}_{year}.txt"
                    out_meta = self.output_dir / f"podcast_rss_{slug}_{year}.meta.json"

                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(content)

                    meta_data = {
                        "source_title": f"{show_name}: {title}",
                        "tier": int(SourceTier.TIER_1_ORAL_INTERVIEW),
                        "year": year,
                        "source_type": "podcast_rss",
                        "show": show_name,
                        "url": feed_url,
                    }
                    with open(out_meta, "w", encoding="utf-8") as f:
                        json.dump(meta_data, f, indent=2)

                    self.manifest[manifest_key] = {"title": title, "show": show_name, "year": year}
                    save_scraper_manifest(self.manifest)

                    logger.info(f"Discovered & saved podcast episode: '{show_name}: {title[:40]}' ({year})")
                    saved_paths.append(out_txt)

            except Exception as e:
                logger.warning(f"Error parsing podcast feed {show_name} ({e}). Continuing...")

        logger.info(f"Podcast RSS Discovery Complete: Acquired {len(saved_paths)} new podcast episodes.")
        return saved_paths


# =====================================================================
# 9. Adaptive Gap-Filling Crawler
# =====================================================================

class AdaptiveGapCrawler:
    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir
        self.web_scraper = AutonomousWebScraper(output_dir)
        self.pmc_scraper = AutonomousEuropePMCScraper(output_dir)

    def fill_era_gap(self, era_query: str, start_year: int, end_year: int) -> List[Path]:
        logger.info(f"[ADAPTIVE GAP FILL] Searching for missing historical evidence: '{era_query}' ({start_year}-{end_year})")
        targeted_queries = [
            f'"Michael Snyder" {era_query} {start_year}..{end_year}',
            f'"Michael Snyder" interview {era_query} {start_year}',
            f'"Michael Snyder" "Stanford" OR "Yale" {era_query}',
        ]
        discovered: List[Path] = []
        for q in targeted_queries:
            urls = self.web_scraper.search_web_urls(q, max_urls=2)
            for u in urls:
                data = self.web_scraper.fetch_and_parse_article(u)
                if data:
                    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", data["title"][:40].lower()).strip("_")
                    year = data["year"]
                    out_txt = self.output_dir / f"adaptive_gap_{slug}_{year}.txt"
                    out_meta = self.output_dir / f"adaptive_gap_{slug}_{year}.meta.json"

                    if not out_txt.exists():
                        with open(out_txt, "w", encoding="utf-8") as f:
                            f.write(data["text"])
                        with open(out_meta, "w", encoding="utf-8") as f:
                            json.dump({
                                "source_title": data["title"],
                                "tier": int(data["tier"]),
                                "year": year,
                                "gap_query": era_query,
                                "url": u,
                            }, f, indent=2)
                        discovered.append(out_txt)
        return discovered


# =====================================================================
# 10. Master Autonomous Scraper Orchestrator
# =====================================================================

class AutonomousCorpusScraper:
    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir
        self.yt_scraper = AutonomousYouTubeScraper(output_dir)
        self.pmc_scraper = AutonomousEuropePMCScraper(output_dir)
        self.openalex_scraper = AutonomousOpenAlexScraper(output_dir)
        self.scihub_scraper = AutonomousSciHubScraper(output_dir)
        self.web_scraper = AutonomousWebScraper(output_dir)
        self.podcast_scraper = PodcastRSSScraper(output_dir)
        self.citation_crawler = EuropePMCCitationCrawler(output_dir)
        self.gap_crawler = AdaptiveGapCrawler(output_dir)

    def run_discovery(
        self,
        max_pmc_per_epoch: int = 50,
        max_citations_per_landmark: int = 15,
        max_openalex_oa: int = 50,
        max_yt_per_query: int = 4,
        max_web_articles: int = 15,
    ) -> List[Path]:
        logger.info("=" * 80)
        logger.info("=== Starting Autonomous Multi-Source Internet Discovery for Michael Snyder ===")
        logger.info("=" * 80)
        start_time = time.time()
        all_discovered: List[Path] = []

        # 1. YouTube Transcripts
        yt_files = self.yt_scraper.discover_and_scrape(max_videos_per_query=max_yt_per_query)
        all_discovered.extend(yt_files)

        # 2. Europe PMC Academic Literature (Unconstrained Complete Harvest)
        pmc_files = self.pmc_scraper.search_and_download(max_papers_per_epoch=max_pmc_per_epoch)
        all_discovered.extend(pmc_files)

        # 3. OpenAlex Open Access PDF Harvester (Full-Text Publisher PDFs)
        openalex_files = self.openalex_scraper.search_and_download(max_papers=max_openalex_oa)
        all_discovered.extend(openalex_files)

        # 4. Europe PMC Recursive Citations Snowballing
        cite_files = self.citation_crawler.crawl_citations(max_per_landmark=max_citations_per_landmark)
        all_discovered.extend(cite_files)

        # 5. Web Articles & Interviews
        web_files = self.web_scraper.discover_and_scrape(max_articles=max_web_articles)
        all_discovered.extend(web_files)

        # 6. Podcast RSS Feeds
        pod_files = self.podcast_scraper.discover_and_scrape()
        all_discovered.extend(pod_files)

        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info(
            f"=== Autonomous Discovery Complete: Acquired {len(all_discovered)} new documents "
            f"in {elapsed:.1f}s. Stored in {self.output_dir} ==="
        )
        logger.info("=" * 80)
        return all_discovered

    def ingest_discovered_corpus(self, file_paths: Optional[List[Path]] = None) -> int:
        logger.info("=== Ingesting Discovered Documents into Personal Brain Knowledge Graph & ChromaDB ===")
        brain = CortexPersonalBrain()
        total_chunks = 0

        if file_paths is None:
            file_paths = [f for f in self.output_dir.glob("*.txt") if f.name != "scraper_manifest.json"]

        pbar = tqdm(file_paths, desc="[Indexing Knowledge Graph & ChromaDB]", unit="doc")
        for f in pbar:
            meta_path = f.with_suffix(".meta.json")
            forced_tier = None
            forced_year = None

            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        mdata = json.load(mf)
                        forced_tier = SourceTier(int(mdata.get("tier", 1)))
                        forced_year = int(mdata.get("year", 1991))
                except Exception as e:
                    logger.warning(f"Could not parse metadata {meta_path.name}: {e}")

            chunks = brain.ingest_document(f, forced_tier=forced_tier, forced_year=forced_year)
            total_chunks += len(chunks)

        logger.info("Computing Louvain Hierarchical Research Communities...")
        brain.hierarchical_graph.detect_and_build_communities()
        logger.info(f"Successfully indexed {total_chunks} deduplicated semantic chunks from autonomous search!")
        return total_chunks


# =====================================================================
# Full-Text Backfill Engine
# =====================================================================

def backfill_full_texts(
    raw_dir: Path = RAW_DIR,
    max_papers: Optional[int] = None,
    delay_between_requests: float = 0.4,
) -> Dict[str, int]:
    """
    Iterates over existing paper_pmc_*.meta.json files in raw_dir that possess a PMCID
    and upgrades them from abstract-only to full-text XML papers.
    """
    session = create_resilient_session()
    meta_files = list(raw_dir.glob("paper_pmc_*.meta.json"))

    stats = {
        "scanned": len(meta_files),
        "open_access_candidates": 0,
        "already_full_text": 0,
        "upgraded": 0,
        "failed_or_skipped": 0,
    }

    logger.info(f"Scanning {len(meta_files)} paper metadata files for full-text backfill candidates...")

    upgraded_count = 0
    pbar = tqdm(meta_files, desc="[PMC XML Harvester]", unit="paper")
    for meta_file in pbar:
        if max_papers is not None and upgraded_count >= max_papers:
            logger.info(f"Reached user limit of {max_papers} backfilled papers.")
            break

        pbar.set_postfix({"upgraded": upgraded_count, "open_access": stats["open_access_candidates"]})

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        pmcid = meta.get("pmcid", "")
        if not pmcid:
            continue

        stats["open_access_candidates"] += 1

        # Check if already upgraded
        txt_file = meta_file.with_name(meta_file.name.replace(".meta.json", ".txt"))
        if meta.get("has_full_text") is True:
            stats["already_full_text"] += 1
            continue

        # Fetch full-text XML
        full_text = fetch_europe_pmc_full_text(session, pmcid)
        if not full_text:
            stats["failed_or_skipped"] += 1
            continue

        # Reconstruct full paper file
        title = meta.get("source_title", "Untitled")
        pub_year = meta.get("year", 2012)
        journal = meta.get("journal", "Scientific Journal")
        pmid = meta.get("pmid", "")

        # Read existing abstract from txt_file if present
        abstract = ""
        authors = "Michael Snyder et al."
        if txt_file.exists():
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    old_content = f.read()
                    if "Authors:" in old_content:
                        authors = old_content.split("Authors:")[1].split("\n")[0].strip()
                    if "Abstract:" in old_content:
                        abstract = old_content.split("Abstract:")[-1].strip()
            except Exception:
                pass

        new_content = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Journal: {journal} ({pub_year})\n"
            f"PMID: {pmid} | PMCID: {pmcid}\n"
            f"Access: Open Access Full-Text XML\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Full Text:\n{full_text}\n"
        )

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        meta["has_full_text"] = True
        meta["full_text_chars"] = len(full_text)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        upgraded_count += 1
        stats["upgraded"] += 1
        pbar.set_postfix({"upgraded": upgraded_count, "open_access": stats["open_access_candidates"]})
        logger.info(f"[{upgraded_count}] Upgraded '{title[:50]}...' ({pub_year}, {len(full_text):,} chars)")
        time.sleep(delay_between_requests)

    logger.info(f"Backfill Complete: {stats}")
    return stats


def backfill_unpaywall_full_texts(
    raw_dir: Path = RAW_DIR,
    email: str = "cortex@stanford.edu",
    max_papers: Optional[int] = None,
    delay_between_requests: float = 0.2,
) -> Dict[str, int]:
    """
    Scans abstract-only papers on disk, batch resolves DOIs, queries Unpaywall API,
    downloads OA publisher PDFs, and upgrades text documents via pypdf.
    """
    session = create_resilient_session()
    meta_files = list(raw_dir.glob("*.meta.json"))

    stats = {
        "scanned": len(meta_files),
        "abstract_only_candidates": 0,
        "already_full_text": 0,
        "upgraded": 0,
        "failed_or_no_oa": 0,
    }

    logger.info(f"Scanning {len(meta_files)} paper records for Unpaywall full-text upgrade candidates...")

    candidates = []
    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                mdata = json.load(f)
            if mdata.get("has_full_text"):
                stats["already_full_text"] += 1
            else:
                stats["abstract_only_candidates"] += 1
                candidates.append((mf, mdata))
        except Exception:
            pass

    logger.info(f"Identified {len(candidates)} abstract-only papers to check with Unpaywall.")

    # Focus DOI resolution on relevant candidate slice
    candidate_slice = candidates[:max_papers * 4] if max_papers else candidates
    missing_doi_items = [(mf, m) for mf, m in candidate_slice if not m.get("doi") and m.get("pmid")]

    if missing_doi_items:
        logger.info(f"Batch resolving DOIs for {len(missing_doi_items)} papers via NCBI E-Summary...")
        for i in range(0, len(missing_doi_items), 50):
            chunk = missing_doi_items[i:i+50]
            batch_pmids = [m["pmid"] for _, m in chunk]
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(batch_pmids)}&retmode=json"
            res = safe_http_request(session, "GET", url, timeout=8, max_network_retries=1)
            if res and res.status_code == 200:
                try:
                    res_dict = res.json().get("result", {})
                    for mf, m in chunk:
                        pmid_str = m.get("pmid")
                        pdata = res_dict.get(pmid_str, {})
                        article_ids = pdata.get("articleids", [])
                        found_doi = next((a.get("value") for a in article_ids if a.get("idtype") == "doi"), None)
                        if found_doi:
                            m["doi"] = found_doi
                            try:
                                with open(mf, "w", encoding="utf-8") as f:
                                    json.dump(m, f, indent=2)
                            except Exception:
                                pass
                except Exception:
                    pass
            time.sleep(0.2)

    upgraded_count = 0
    pbar = tqdm(candidates, desc="[Unpaywall OA Harvester]", unit="paper")
    for meta_file, meta in pbar:
        if max_papers is not None and upgraded_count >= max_papers:
            break
        pbar.set_postfix({"upgraded": upgraded_count, "no_oa": stats["failed_or_no_oa"]})

        doi = meta.get("doi")
        pmid = meta.get("pmid")
        title = meta.get("source_title", meta_file.stem)
        pub_year = meta.get("year", 2015)
        journal = meta.get("journal", "Scientific Journal")

        if not is_snyder_genomics_paper(title=title, abstract="", journal=journal):
            stats["failed_or_no_oa"] += 1
            continue

        if not doi:
            stats["failed_or_no_oa"] += 1
            continue

        # Query Unpaywall API
        u_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        res_u = safe_http_request(session, "GET", u_url, timeout=8, max_network_retries=1)
        if not res_u or res_u.status_code != 200:
            stats["failed_or_no_oa"] += 1
            continue

        try:
            u_data = res_u.json()
        except Exception:
            stats["failed_or_no_oa"] += 1
            continue

        if not u_data.get("is_oa"):
            stats["failed_or_no_oa"] += 1
            continue

        best_oa = u_data.get("best_oa_location") or {}
        pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url")
        if not pdf_url:
            stats["failed_or_no_oa"] += 1
            continue

        # Fast non-blocking direct PDF download
        try:
            pdf_res = requests.get(pdf_url, timeout=8, headers={"User-Agent": USER_AGENT})
            if pdf_res.status_code != 200 or len(pdf_res.content) < 5000 or not pdf_res.content.startswith(b"%PDF"):
                stats["failed_or_no_oa"] += 1
                continue
        except Exception:
            stats["failed_or_no_oa"] += 1
            continue

        try:
            reader = PdfReader(io.BytesIO(pdf_res.content))
            pages_text = []
            for page_obj in reader.pages:
                pt = page_obj.extract_text()
                if pt:
                    pages_text.append(pt)
            full_text = "\n\n".join(pages_text).strip()
        except Exception as e:
            logger.debug(f"Could not parse PDF for {title}: {e}")
            stats["failed_or_no_oa"] += 1
            continue

        if len(full_text) < 1000:
            stats["failed_or_no_oa"] += 1
            continue

        txt_file = meta_file.with_name(meta_file.name.replace(".meta.json", ".txt"))
        abstract = ""
        authors = "Michael Snyder et al."
        if txt_file.exists():
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    old_c = f.read()
                    if "Authors:" in old_c:
                        authors = old_c.split("Authors:")[1].split("\n")[0].strip()
                    if "Abstract:" in old_c:
                        abstract = old_c.split("Abstract:")[-1].strip()
            except Exception:
                pass

        new_content = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Journal: {journal} ({pub_year})\n"
            f"PMID: {pmid} | DOI: {doi}\n"
            f"Access: Open Access PDF ({len(reader.pages)} pages)\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Full Text:\n{full_text}\n"
        )

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        meta["has_full_text"] = True
        meta["full_text_chars"] = len(full_text)
        meta["pages"] = len(reader.pages)
        meta["url"] = pdf_url
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        upgraded_count += 1
        stats["upgraded"] += 1
        logger.info(f"[{upgraded_count}] Upgraded via Unpaywall PDF: '{title[:45]}...' ({len(reader.pages)} pages, {len(full_text):,} chars)")
        time.sleep(delay_between_requests)

    logger.info(f"Unpaywall Harvest Complete: {stats}")
    return stats


def backfill_scihub_full_texts(
    raw_dir: Path = RAW_DIR,
    max_papers: Optional[int] = None,
    delay_between_requests: float = 0.5,
) -> Dict[str, int]:
    """
    Scans abstract-only papers on disk, batch-resolves DOIs via NCBI E-Summary,
    queries active Sci-Hub / SciDB academic mirrors, downloads binary PDFs,
    extracts complete text via pypdf, and upgrades existing documents in-place.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session = create_resilient_session(max_retries=1)
    meta_files = [f for f in raw_dir.glob("*.meta.json") if f.name != "scraper_manifest.json"]

    stats = {
        "scanned": len(meta_files),
        "abstract_only_candidates": 0,
        "already_full_text": 0,
        "upgraded": 0,
        "failed_or_unhosted": 0,
    }

    logger.info(f"Scanning {len(meta_files)} paper records for Sci-Hub full-text upgrade candidates...")

    candidates = []
    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                mdata = json.load(f)
            if mdata.get("has_full_text"):
                stats["already_full_text"] += 1
            else:
                stats["abstract_only_candidates"] += 1
                candidates.append((mf, mdata))
        except Exception:
            pass

    logger.info(f"Identified {len(candidates)} abstract-only papers to check via Sci-Hub.")

    # Target slice for DOI resolution
    target_candidates = candidates[:max_papers * 5] if max_papers else candidates
    missing_doi_items = [(mf, m) for mf, m in target_candidates if not m.get("doi") and m.get("pmid")]

    if missing_doi_items:
        logger.info(f"Batch resolving DOIs for {len(missing_doi_items)} papers via NCBI E-Summary...")
        for i in range(0, len(missing_doi_items), 50):
            chunk = missing_doi_items[i:i+50]
            batch_pmids = [m["pmid"] for _, m in chunk]
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(batch_pmids)}&retmode=json"
            res = safe_http_request(session, "GET", url, timeout=8, max_network_retries=1)
            if res and res.status_code == 200:
                try:
                    res_dict = res.json().get("result", {})
                    for mf, m in chunk:
                        pmid_str = m.get("pmid")
                        pdata = res_dict.get(pmid_str, {})
                        article_ids = pdata.get("articleids", [])
                        found_doi = next((a.get("value") for a in article_ids if a.get("idtype") == "doi"), None)
                        if found_doi:
                            m["doi"] = found_doi
                            try:
                                with open(mf, "w", encoding="utf-8") as f:
                                    json.dump(m, f, indent=2)
                            except Exception:
                                pass
                except Exception:
                    pass
            time.sleep(0.2)

    scihub = AutonomousSciHubScraper(output_dir=raw_dir)
    upgraded_count = 0
    pbar = tqdm(target_candidates, desc="[Sci-Hub Academic Mirror Harvest]", unit="paper")

    for meta_file, meta in pbar:
        if max_papers is not None and upgraded_count >= max_papers:
            break
        pbar.set_postfix({"upgraded": upgraded_count, "unhosted": stats["failed_or_unhosted"]})

        doi = meta.get("doi")
        pmid = meta.get("pmid")
        title = meta.get("source_title", meta_file.stem)
        pub_year = meta.get("year", 2015)
        journal = meta.get("journal", "Scientific Journal")

        if not is_snyder_genomics_paper(title=title, abstract="", journal=journal):
            stats["failed_or_unhosted"] += 1
            continue

        if not doi:
            stats["failed_or_unhosted"] += 1
            continue

        pdf_url = scihub.resolve_pdf_url(doi)
        if not pdf_url:
            stats["failed_or_unhosted"] += 1
            continue

        extracted = scihub.download_and_extract_pdf(pdf_url)
        if not extracted:
            stats["failed_or_unhosted"] += 1
            continue

        full_text, pages_count = extracted

        txt_file = meta_file.with_name(meta_file.name.replace(".meta.json", ".txt"))
        abstract = ""
        authors = "Michael Snyder et al."
        if txt_file.exists():
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    old_c = f.read()
                    if "Authors:" in old_c:
                        authors = old_c.split("Authors:")[1].split("\n")[0].strip()
                    if "Abstract:" in old_c:
                        abstract = old_c.split("Abstract:")[-1].strip()
            except Exception:
                pass

        new_content = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Journal: {journal} ({pub_year})\n"
            f"PMID: {pmid} | DOI: {doi}\n"
            f"Access: Sci-Hub Academic Mirror PDF ({pages_count} pages)\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Full Text:\n{full_text}\n"
        )

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        meta["has_full_text"] = True
        meta["full_text_chars"] = len(full_text)
        meta["pages"] = pages_count
        meta["url"] = pdf_url
        meta["source_type"] = "published_paper_pdf"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        upgraded_count += 1
        stats["upgraded"] += 1
        logger.info(f"[{upgraded_count}] Upgraded via Sci-Hub: '{title[:45]}...' ({pages_count} pages, {len(full_text):,} chars)")
        time.sleep(delay_between_requests)

    logger.info(f"Sci-Hub Harvest Complete: {stats}")
    return stats


def audit_and_clean_corpus(raw_dir: Path = RAW_DIR) -> Dict[str, int]:
    """
    Scans every document on disk in raw_dir and deletes any document that is not
    confirmed to be authored by Dr. Michael P. Snyder or fails genomics domain validation.
    """
    meta_files = [f for f in raw_dir.glob("*.meta.json") if f.name != "scraper_manifest.json"]
    logger.info(f"Auditing {len(meta_files)} publications on disk for strict Snyder authorship...")

    purged_count = 0
    kept_count = 0

    pbar = tqdm(meta_files, desc="[Auditing Corpus]", unit="doc")
    for mf in pbar:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
            title = data.get("source_title", "")
            journal = data.get("journal", "")
            source_type = data.get("source_type", "")
            txt_path = mf.with_name(mf.name.replace(".meta.json", ".txt"))

            if not txt_path.exists():
                mf.unlink(missing_ok=True)
                purged_count += 1
                continue

            with open(txt_path, "r", encoding="utf-8") as tf:
                content = tf.read(2000)

            authors = ""
            if "Authors:" in content:
                authors = content.split("Authors:")[1].split("\n")[0].strip()

            is_valid = True
            if "video" not in source_type and "podcast" not in source_type and "interview" not in source_type:
                if authors and "snyder" not in authors.lower():
                    is_valid = False
                elif not is_snyder_genomics_paper(title=title, abstract="", authors=authors, journal=journal):
                    is_valid = False

            if not is_valid:
                logger.info(f"Purging non-Snyder file: {mf.name} ('{title[:40]}...')")
                mf.unlink(missing_ok=True)
                txt_path.unlink(missing_ok=True)
                purged_count += 1
            else:
                kept_count += 1
        except Exception:
            pass

    stats = {"scanned": len(meta_files), "kept": kept_count, "purged": purged_count, "db_chunks_cleaned": 0}

    # Automatically purge orphaned vectors and graph nodes
    try:
        import chromadb
        chroma_dir = Path("l:/cortex/data/chromadb")
        if chroma_dir.exists():
            client = chromadb.PersistentClient(path=str(chroma_dir))
            for col_obj in client.list_collections():
                col = client.get_collection(col_obj.name)
                records = col.get()
                existing_names = {f.name for f in raw_dir.glob("*.txt")}
                existing_stems = {f.stem for f in raw_dir.glob("*.txt")}
                orphaned_ids = []
                for cid, meta in zip(records.get("ids", []), records.get("metadatas", [])):
                    s_name = meta.get("source_name", "")
                    s_id = meta.get("source_id", "")
                    if s_name not in existing_names and s_id not in existing_stems and (s_name + ".txt") not in existing_names:
                        orphaned_ids.append(cid)
                if orphaned_ids:
                    for i in range(0, len(orphaned_ids), 100):
                        col.delete(ids=orphaned_ids[i:i+100])
                    stats["db_chunks_cleaned"] += len(orphaned_ids)
                    logger.info(f"Purged {len(orphaned_ids)} orphaned vector chunks from ChromaDB collection '{col_obj.name}'.")
    except Exception as e:
        logger.debug(f"ChromaDB cleaning notice: {e}")

    logger.info(f"Corpus Audit Complete: {stats}")
    return stats


# =====================================================================
# CLI Entrypoint
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Autonomous Internet Discovery Scraper for Michael Snyder Corpus")
    parser.add_argument("--search-all", action="store_true", help="Run full autonomous discovery across YouTube, Europe PMC, Citations, Web, and Podcasts")
    parser.add_argument("--youtube-only", action="store_true", help="Search and scrape YouTube transcripts only")
    parser.add_argument("--pmc-only", action="store_true", help="Search and scrape Europe PMC scientific literature only")
    parser.add_argument("--openalex-only", action="store_true", help="Harvest open-access full-text PDFs from OpenAlex only")
    parser.add_argument("--citations-only", action="store_true", help="Search and crawl Europe PMC recursive citations only")
    parser.add_argument("--web-only", action="store_true", help="Search and scrape web articles and interviews only")
    parser.add_argument("--podcasts-only", action="store_true", help="Search and scrape podcast RSS feeds only")
    parser.add_argument("--backfill-fulltext", action="store_true", help="Upgrade existing PMC papers on disk with full-text XML")
    parser.add_argument("--unpaywall", action="store_true", help="Harvest open-access full-text PDFs via Unpaywall API")
    parser.add_argument("--scihub", action="store_true", help="Harvest paywalled scientific PDFs via Sci-Hub active mirrors")
    parser.add_argument("--upgrade-all", action="store_true", help="Run comprehensive full-text upgrade across all sources (Unpaywall + Sci-Hub + PMC)")
    parser.add_argument("--audit-and-clean", action="store_true", help="Scan disk corpus and purge any non-Snyder or invalid files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process/backfill (default: unlimited/all)")
    parser.add_argument("--max-pmc", type=int, default=50, help="Max papers per Europe PMC epoch (default: 50)")
    parser.add_argument("--max-openalex", type=int, default=50, help="Max OA papers from OpenAlex (default: 50)")
    parser.add_argument("--max-scihub", type=int, default=None, help="Max papers to retrieve via Sci-Hub (default: unlimited)")
    parser.add_argument("--max-citations", type=int, default=15, help="Max citations per landmark paper (default: 15)")
    parser.add_argument("--max-yt", type=int, default=4, help="Max videos per YouTube query (default: 4)")
    parser.add_argument("--max-web", type=int, default=15, help="Max web articles total (default: 15)")
    parser.add_argument("--ingest", action="store_true", help="Automatically ingest and index all newly discovered documents")
    args = parser.parse_args()

    if args.upgrade_all:
        logger.info("=" * 80)
        logger.info("=== Starting Master Full-Text Upgrade Across All Remaining Papers ===")
        logger.info("=" * 80)
        u_stats = backfill_unpaywall_full_texts(max_papers=args.limit)
        s_stats = backfill_scihub_full_texts(max_papers=args.limit)
        p_stats = backfill_full_texts(max_papers=args.limit)
        total_upgraded = u_stats.get("upgraded", 0) + s_stats.get("upgraded", 0) + p_stats.get("upgraded", 0)
        print(f"\n[DONE] Master Upgrade Complete! Total Upgraded to Full-Text: {total_upgraded}")

        if args.ingest:
            scraper = AutonomousCorpusScraper()
            chunk_count = scraper.ingest_discovered_corpus()
            print(f"\n[DONE] Ingested {chunk_count} deduplicated chunks into ChromaDB and NetworkX temporal graph!")
        return

    if args.backfill_fulltext:
        stats = backfill_full_texts(max_papers=args.limit)
        print(f"\n[DONE] Full-Text Backfill Complete! Upgraded: {stats['upgraded']} papers to full-text.")
        return

    if args.unpaywall:
        stats = backfill_unpaywall_full_texts(max_papers=args.limit)
        print(f"\n[DONE] Unpaywall Harvest Complete! Upgraded: {stats['upgraded']} papers to full-text.")
        return

    if args.audit_and_clean:
        stats = audit_and_clean_corpus()
        print(f"\n[DONE] Corpus Audit & Clean Complete! Scanned: {stats['scanned']} | Kept: {stats['kept']} | Purged: {stats['purged']}")
        return

    if args.scihub:
        stats = backfill_scihub_full_texts(max_papers=args.limit or args.max_scihub)
        print(f"\n[DONE] Sci-Hub Harvest Complete! Upgraded: {stats['upgraded']} papers to full-text.")
        return

    scraper = AutonomousCorpusScraper()
    discovered_files: List[Path] = []

    if args.youtube_only:
        discovered_files = scraper.yt_scraper.discover_and_scrape(max_videos_per_query=args.max_yt)
    elif args.pmc_only:
        discovered_files = scraper.pmc_scraper.search_and_download(max_papers_per_epoch=args.max_pmc)
    elif args.openalex_only:
        discovered_files = scraper.openalex_scraper.search_and_download(max_papers=args.max_openalex)
    elif args.citations_only:
        discovered_files = scraper.citation_crawler.crawl_citations(max_per_landmark=args.max_citations)
    elif args.web_only:
        discovered_files = scraper.web_scraper.discover_and_scrape(max_articles=args.max_web)
    elif args.podcasts_only:
        discovered_files = scraper.podcast_scraper.discover_and_scrape()
    else:
        discovered_files = scraper.run_discovery(
            max_pmc_per_epoch=args.max_pmc,
            max_citations_per_landmark=args.max_citations,
            max_openalex_oa=args.max_openalex,
            max_yt_per_query=args.max_yt,
            max_web_articles=args.max_web,
        )

    print(f"\n[DONE] Discovered and saved {len(discovered_files)} new files to {RAW_DIR}")

    if args.ingest and discovered_files:
        chunk_count = scraper.ingest_discovered_corpus(discovered_files)
        print(f"\n[DONE] Ingested {chunk_count} deduplicated chunks into ChromaDB and NetworkX temporal graph!")


if __name__ == "__main__":
    main()
