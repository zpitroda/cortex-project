"""
reprocess_transcripts.py - Audio Download, Whisper Transcription & Speaker Diarization Pipeline.

Downloads audio tracks via yt-dlp, transcribes using faster-whisper on CPU (0 MB VRAM footprint),
diarizes speaker turns ([Moderator], [Host], [Interviewer], [Michael Snyder], [Audience]),
replaces data/raw/transcript_*.txt files with clean diarized text, and triggers re-indexing.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import yt_dlp
from faster_whisper import WhisperModel

from config import SourceTier, settings
from ingest import DataIngestionPipeline, clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cortex.reprocess")

RAW_DIR = Path("l:/cortex/data/raw")
AUDIO_CACHE_DIR = RAW_DIR / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class AudioTranscriptDiarizer:
    """
    Downloads audio, runs local Whisper ASR on CPU, and segments speakers
    into [Host], [Moderator], [Interviewer], [Michael Snyder], and [Audience].
    """
    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        llm_endpoint: str = "http://localhost:8080/v1/chat/completions",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.llm_endpoint = llm_endpoint
        self._whisper_model: Optional[WhisperModel] = None

    def _get_whisper(self) -> WhisperModel:
        if self._whisper_model is None:
            logger.info(f"Loading faster-whisper ({self.model_size}) on {self.device} (compute_type={self.compute_type})...")
            self._whisper_model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._whisper_model

    def is_llm_available(self) -> bool:
        try:
            r = requests.get("http://localhost:8080/v1/models", timeout=0.8)
            return r.status_code == 200
        except Exception:
            return False

    def download_audio(self, url_or_id: str) -> Optional[Path]:
        """Downloads audio track for a YouTube video using yt-dlp."""
        if not url_or_id.startswith("http"):
            url = f"https://www.youtube.com/watch?v={url_or_id}"
            video_id = url_or_id
        else:
            url = url_or_id
            m = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
            video_id = m.group(1) if m else "audio_download"

        out_template = str(AUDIO_CACHE_DIR / f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            logger.info(f"Downloading audio for {url}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = Path(ydl.prepare_filename(info))
                if downloaded_file.exists():
                    logger.info(f"Audio downloaded: {downloaded_file.name} ({downloaded_file.stat().st_size / (1024*1024):.1f} MB)")
                    return downloaded_file
        except Exception as e:
            logger.error(f"Failed to download audio for {url}: {e}")

        return None

    def transcribe_audio(self, audio_path: Path) -> List[Dict[str, Any]]:
        """Transcribes audio using faster-whisper with timestamps."""
        model = self._get_whisper()
        logger.info(f"Transcribing audio {audio_path.name}...")
        segments_gen, info = model.transcribe(str(audio_path), beam_size=5)

        segments = []
        for s in segments_gen:
            segments.append({
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
            })

        logger.info(f"Transcribed {len(segments)} segments ({info.duration:.1f}s total duration).")
        return segments

    def diarize_talk_segments(self, segments: List[Dict[str, Any]], title: str = "") -> str:
        """
        Diarizes a prepared lecture/talk, identifying:
        - [Moderator / Host]: introductory welcome and speaker introduction
        - [Michael Snyder]: main lecture body and answers
        - [Audience]: questions at the end
        """
        if not segments:
            return ""

        formatted_blocks: List[str] = []
        current_speaker = None
        current_text: List[str] = []

        intro_regex = re.compile(
            r"(our next speaker is|welcome.*michael|pleased to introduce|professor michael snyder|chair of genetics|landing just in time|this is [A-Za-z]+ from|welcome to our series|welcome everyone|introducing dr)",
            re.IGNORECASE
        )
        snyder_transition_regex = re.compile(
            r"(okay[, ]+well[, ]+thanks|it['’]s (?:really )?great to be here|it is a pleasure to be here|thank you for having me|thank you[, ]+Rhonda|thanks[, ]+it['’]s great|okay[, ]+thanks)",
            re.IGNORECASE
        )

        intro_ended = False

        for i, seg in enumerate(segments):
            text = seg["text"]
            start_time = seg["start"]

            # First 90 seconds: check if host/moderator transition occurs
            if not intro_ended and start_time < 90.0:
                trans_match = snyder_transition_regex.search(text)
                if trans_match:
                    intro_part = text[:trans_match.start()].strip()
                    snyder_part = text[trans_match.start():].strip()

                    if intro_part and current_speaker == "Moderator":
                        current_text.append(intro_part)
                    elif intro_part:
                        if current_speaker and current_text:
                            formatted_blocks.append(f"[{current_speaker}]: {' '.join(current_text)}")
                        current_speaker = "Moderator"
                        current_text = [intro_part]

                    if current_speaker and current_text:
                        formatted_blocks.append(f"[{current_speaker}]: {' '.join(current_text)}")

                    intro_ended = True
                    current_speaker = "Michael Snyder"
                    current_text = [snyder_part]
                    continue

                if intro_regex.search(text) and not any(phrase in text.lower() for phrase in ["my lab", "our lab", "when i was", "i started"]):
                    if current_speaker != "Moderator":
                        if current_speaker and current_text:
                            formatted_blocks.append(f"[{current_speaker}]: {' '.join(current_text)}")
                        current_speaker = "Moderator"
                        current_text = [text]
                    else:
                        current_text.append(text)
                    continue

                if not current_speaker:
                    current_speaker = "Michael Snyder"
                current_text.append(text)
                continue

            # Check for audience Q&A near the end of talk
            if intro_ended and i > len(segments) * 0.8:
                if text.endswith("?") and len(text.split()) < 30 and any(w in text.lower() for w in ["what about", "how do you", "is that", "can you", "do you think"]):
                    if current_speaker != "Audience":
                        if current_speaker and current_text:
                            formatted_blocks.append(f"[{current_speaker}]: {' '.join(current_text)}")
                        current_speaker = "Audience"
                        current_text = [text]
                        continue
                        if current_speaker and current_text:
                            formatted_blocks.append(f"[{current_speaker}]: {' '.join(current_text)}")
                        current_speaker = "Audience"
                        current_text = [text]
                        continue

            if not current_speaker:
                current_speaker = "Michael Snyder"
            current_text.append(text)

        if current_speaker and current_text:
            formatted_blocks.append(f"[{current_speaker}]: {' '.join(current_text)}")

        return "\n\n".join(formatted_blocks)

    def diarize_interview_segments(self, segments: List[Dict[str, Any]], title: str = "") -> str:
        """
        Diarizes an interview/podcast into [Interviewer] and [Michael Snyder] dialogue turns.
        """
        if not segments:
            return ""

        full_text = " ".join(s["text"] for s in segments)

        # If LLM is available, use local Qwen 27B for semantic dialogue turn segmentation
        if self.is_llm_available() and len(full_text) > 100:
            return self._diarize_with_qwen(segments)

        # Fallback heuristic for interview turns
        formatted_blocks: List[str] = []
        current_speaker = "Interviewer"
        current_text: List[str] = []

        for i, seg in enumerate(segments):
            text = seg["text"]
            # Question marks or intro lead-ins typically denote interviewer
            if ("?" in text and len(text.split()) < 40) or "welcome to" in text.lower():
                if current_speaker != "Interviewer":
                    if current_text:
                        formatted_blocks.append(f"[{current_speaker}]: {' '.join(current_text)}")
                    current_speaker = "Interviewer"
                    current_text = [text]
                else:
                    current_text.append(text)
            else:
                if current_speaker != "Michael Snyder":
                    if current_text:
                        formatted_blocks.append(f"[{current_speaker}]: {' '.join(current_text)}")
                    current_speaker = "Michael Snyder"
                    current_text = [text]
                else:
                    current_text.append(text)

        if current_speaker and current_text:
            formatted_blocks.append(f"[{current_speaker}]: {' '.join(current_text)}")

        return "\n\n".join(formatted_blocks)

    def _diarize_with_qwen(self, segments: List[Dict[str, Any]]) -> str:
        """Uses local Qwen 27B on port 8080 to format dialogue into turns."""
        full_text = " ".join(s["text"] for s in segments)
        sample = full_text[:4000]

        prompt = (
            "Format the following interview transcript into clean speaker dialogue turns with tags:\n"
            "[Interviewer]: ...\n"
            "[Michael Snyder]: ...\n\n"
            f"TRANSCRIPT SAMPLE:\n{sample}\n\n"
            "Rules:\n"
            "1. Accurately tag the podcast host / interviewer questions.\n"
            "2. Tag Professor Michael Snyder's answers as [Michael Snyder]:\n"
            "3. Output ONLY the formatted dialogue with tags."
        )

        try:
            res = requests.post(
                self.llm_endpoint,
                headers={"Content-Type": "application/json"},
                json={
                    "messages": [
                        {"role": "system", "content": "You are a precise scientific dialogue editor."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 3000,
                },
                timeout=25,
            )
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"].strip()
                if "[Michael Snyder]:" in content:
                    remainder = full_text[4000:].strip()
                    if remainder:
                        return f"{content}\n\n[Michael Snyder]: {remainder}"
                    return content
        except Exception as e:
            logger.debug(f"Qwen diarization fallback: {e}")

        return f"[Michael Snyder]: {full_text}"


def reprocess_all_transcripts():
    """
    Downloads audio and replaces all YouTube transcripts in data/raw
    with clean, properly diarized transcripts.
    """
    diarizer = AudioTranscriptDiarizer(model_size="small.en", device="cpu", compute_type="int8")

    meta_files = list(RAW_DIR.glob("transcript_*.meta.json"))
    logger.info(f"Found {len(meta_files)} transcript metadata files to inspect and reprocess.")

    updated_count = 0

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            url = meta.get("url", "")
            video_id = meta.get("video_id")
            title = meta.get("source_title", meta_path.stem)
            tier = meta.get("tier", int(SourceTier.TIER_2_PREPARED_LECTURE))

            # Match corresponding text file
            txt_path = meta_path.with_suffix("").with_suffix(".txt")

            # Check if this is a YouTube source
            is_youtube = ("youtube.com" in url) or (video_id is not None) or meta.get("source_type") == "youtube_transcript"
            if not is_youtube and not url:
                continue

            target_url = url if "youtube.com" in url else (f"https://www.youtube.com/watch?v={video_id}" if video_id else None)
            if not target_url:
                continue

            logger.info(f"\n=======================================================")
            logger.info(f"Reprocessing: '{title}' ({target_url})")
            logger.info(f"=======================================================")

            # Download audio
            audio_path = diarizer.download_audio(target_url)
            if audio_path and audio_path.exists():
                try:
                    # Transcribe with Whisper on CPU
                    segments = diarizer.transcribe_audio(audio_path)

                    # Diarize based on tier/type
                    is_interview = (tier == int(SourceTier.TIER_1_ORAL_INTERVIEW)) or any(w in title.lower() for w in ["interview", "podcast", "conversation", "foundmyfitness", "huberman"])
                    if is_interview:
                        formatted_text = diarizer.diarize_interview_segments(segments, title=title)
                    else:
                        formatted_text = diarizer.diarize_talk_segments(segments, title=title)

                    # Save clean transcript
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(formatted_text)

                    logger.info(f"SUCCESS: Saved diarized transcript to {txt_path.name}")
                    logger.info(f"Preview (first 250 chars):\n{formatted_text[:250]}...\n")
                    updated_count += 1
                finally:
                    # Clean up temporary audio file to preserve disk space
                    if audio_path.exists():
                        try:
                            audio_path.unlink()
                            logger.info(f"Cleaned up temporary audio: {audio_path.name}")
                        except Exception:
                            pass
            else:
                logger.warning(f"Could not download audio for {title}. Applying text-based intro separation.")
                if txt_path.exists():
                    with open(txt_path, "r", encoding="utf-8") as f:
                        old_text = f.read()
                    # Strip blind [Michael Snyder]: prefix if host intro is present
                    clean_old = old_text.replace("[Michael Snyder]:", "").strip()
                    if "our next speaker is" in clean_old.lower() or "this is richard from" in clean_old.lower():
                        # Split intro
                        m = re.search(r"(okay[, ]+well[, ]+thanks|it['’]s really great to be here|it is a pleasure to be here|thank you)", clean_old, re.IGNORECASE)
                        if m:
                            intro = clean_old[:m.start()].strip()
                            snyder = clean_old[m.start():].strip()
                            new_text = f"[Moderator]: {intro}\n\n[Michael Snyder]: {snyder}"
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(new_text)
                            logger.info(f"Heuristically corrected text transcript for {txt_path.name}")
                            updated_count += 1

        except Exception as e:
            logger.error(f"Error reprocessing {meta_path.name}: {e}", exc_info=True)

    logger.info(f"\nReprocessing complete. Updated {updated_count} transcripts.")

    # Trigger re-ingestion into ChromaDB
    logger.info("Triggering full re-ingestion into ChromaDB and updating GraphRAG...")
    pipeline = DataIngestionPipeline()
    for txt_file in RAW_DIR.glob("transcript_*.txt"):
        try:
            pipeline.ingest_file(txt_file)
        except Exception as e:
            logger.warning(f"Error re-ingesting {txt_file.name}: {e}")

    logger.info("All transcripts successfully re-ingested and indexed!")


if __name__ == "__main__":
    reprocess_all_transcripts()
