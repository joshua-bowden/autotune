"""
Search module for the radio archiver system.

This module provides a command-line interface for searching stored radio
stories using semantic similarity. It converts user queries into embeddings,
finds the top k matches, stitches their audio with beeps, and writes a report.

Usage:
    python search.py <top_k> <query text>

Example:
    python search.py 5 climate change policy
"""

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import database
import config
import json
from pydub import AudioSegment

from utils import get_embedding, setup_logging


logger = setup_logging(__name__)


# Clipping constants
DEFAULT_TOP_K = config.SEARCH_DEFAULT_TOP_K
EXCERPT_LENGTH = config.SEARCH_EXCERPT_LENGTH
CLIP_PADDING_MS = config.CLIP_PADDING_MS


def get_audio_segment(
    start_samples: Optional[int] = None,
    end_samples: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Optional[AudioSegment]:
    """
    Loads relevant MP3 chunks, concatenates them, and returns the clipped segment
    as an AudioSegment (no file written). Uses sample offsets and optional session_id.
    Returns None if the range cannot be satisfied.
    """
    try:
        if start_samples is None or end_samples is None:
            logger.error("start_samples and end_samples are required for clipping")
            return None

        padding_ms = CLIP_PADDING_MS
        audio_files = sorted(list(config.AUDIO_DIR.glob("kqed_*.mp3")), reverse=True)
        if not audio_files:
            logger.warning("No audio files found for clipping")
            return None

        relevant_files = []

        def sessions_match(file_session: str, target: str) -> bool:
            if file_session == target:
                return True
            if file_session.startswith(target + "_"):
                return True
            if target.startswith(file_session + "_"):
                return True
            return False

        target_session = session_id
        if target_session and not target_session.startswith("kqed_"):
            target_session = "kqed_" + target_session

        for file_path in audio_files:
            match = re.search(r'(kqed_\d{8}_\d{6}(?:_\d{6})?)_(\d{12})', file_path.stem)
            if not match:
                continue
            session_str = match.group(1)
            offset_samples = int(match.group(2))
            if target_session and not sessions_match(session_str, target_session):
                continue
            padding_samples = int((padding_ms / 1000.0) * config.VAD_SAMPLING_RATE)
            s_start = start_samples - padding_samples
            s_end = end_samples + padding_samples
            f_start = offset_samples
            f_end = offset_samples + config.SAMPLES_PER_ARCHIVE
            if f_start < s_end and f_end > s_start:
                if not target_session:
                    target_session = session_str
                relevant_files.append((offset_samples, file_path))

        if not relevant_files:
            logger.warning(
                f"No audio files found for range samples=[{start_samples}, {end_samples}] session_id={session_id}"
            )
            return None

        relevant_files.sort(key=lambda x: x[0])
        combined = AudioSegment.empty()
        for _, file_path in relevant_files:
            segment = AudioSegment.from_mp3(str(file_path))
            combined += segment

        first_file_sample_offset = relevant_files[0][0]
        padding_samples = int((padding_ms / 1000.0) * config.VAD_SAMPLING_RATE)
        final_start_samples = (start_samples - padding_samples) - first_file_sample_offset
        final_end_samples = (end_samples + padding_samples) - first_file_sample_offset
        final_start = max(0, int((final_start_samples / config.VAD_SAMPLING_RATE) * 1000))
        final_end = min(len(combined), int((final_end_samples / config.VAD_SAMPLING_RATE) * 1000))

        if final_end <= final_start:
            logger.error(f"Invalid clip range: start={final_start}ms, end={final_end}ms")
            return None

        if final_end - final_start < 100:
            final_end = final_start + 100
            if final_end > len(combined):
                final_start = max(0, len(combined) - 100)
                final_end = len(combined)

        return combined[final_start:final_end]
    except Exception as e:
        logger.error(f"Error clipping audio: {e}")
        return None


def extract_audio_segment(story_id: int, query: str = "clip", start_samples: Optional[int] = None, end_samples: Optional[int] = None, session_id: Optional[str] = None) -> Optional[str]:
    """
    Finds relevant MP3 chunks, concatenates them, and clips to the exact story range.
    Uses sample offsets and an optional session_id for 100% timing accuracy.
    """
    try:
        safe_query = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        clipped = get_audio_segment(start_samples=start_samples, end_samples=end_samples, session_id=session_id)
        if clipped is None:
            return None
        results_dir = config.RESULTS_DIR
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / f"{safe_query}.mp3"
        clipped.export(
            str(output_path),
            format="mp3",
            bitrate="128k",
            parameters=["-q:a", "2"]
        )
        if not output_path.exists() or output_path.stat().st_size == 0:
            logger.error(f"Failed to create valid MP3 file: {output_path}")
            return None
        return str(output_path)
    except Exception as e:
        logger.error(f"Error clipping audio: {e}")
        return None

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results for display.
    
    Args:
        results: List of story dictionaries from database search
        
    Returns:
        Formatted string for display
    """
    if not results:
        return "No results found."
    
    output_lines = [f"\nFound {len(results)} results:\n"]
    
    for i, result in enumerate(results, 1):
        summary = result.get("summary", "No summary")
        distance = result.get("distance", 0.0)
        transcript = result.get("transcript", "")
        audio_path = result.get("audio_path", "{}")
        
        # Parse audio metadata
        try:
            audio_metadata = json.loads(audio_path)
            start_iso = audio_metadata.get("start_time", "Unknown")
            duration = audio_metadata.get("duration", 0)
            
            # Format timestamp for display
            if start_iso != "Unknown":
                try:
                    dt = datetime.fromisoformat(start_iso)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = start_iso
            else:
                time_str = "Unknown"
            
            audio_info = f"{time_str} ({duration:.1f}s)"
        except (json.JSONDecodeError, TypeError):
            audio_info = "Metadata unavailable"
        
        # Excerpt from transcript (plain story text; or legacy JSON if present)
        excerpt = transcript[:EXCERPT_LENGTH]
        if len(transcript) > EXCERPT_LENGTH:
            excerpt += "..."
        output_lines.append(f"{i}. {summary}")
        output_lines.append(f"   Similarity: {1 - distance:.4f}")  # Convert distance to similarity
        output_lines.append(f"   Audio: {audio_info}")
        output_lines.append(f"   Excerpt: {excerpt}")
        output_lines.append("-" * 60)
    
    return "\n".join(output_lines)


def search_stories(query: str, top_k: int = DEFAULT_TOP_K) -> None:
    """
    Search for stories matching the query, stitch top k clips with beeps, and write a report.
    """
    logger.info(f"Searching for: '{query}' (top_k={top_k})")

    embedding = get_embedding(query)
    if embedding is None:
        logger.error("Failed to generate embedding for query")
        print("Error: Could not process your query. Please try again.")
        return

    results = database.search_stories(embedding, top_k=top_k)
    if not results:
        print("No results found.")
        return

    # Build safe base filename from query and timestamp
    safe_query = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")[:40]
    timestamp = datetime.now().strftime("%d_%m_%Y_%H%M")
    base_name = f"search_k{top_k}_{safe_query}_{timestamp}"

    # Extract audio for each result and collect clip paths
    clips = []
    for i, result in enumerate(results, 1):
        summary = result.get("summary", "No summary")
        audio_path = result.get("audio_path", "{}")
        story_id = result.get("id")
        try:
            audio_meta = json.loads(audio_path)
            print(f"[{i}] Clipping audio for: {summary[:50]}...")
            clip_path = extract_audio_segment(
                story_id,
                query=f"{base_name}_result{i}_s{story_id}",
                start_samples=audio_meta.get("start_samples"),
                end_samples=audio_meta.get("end_samples"),
                session_id=audio_meta.get("session_id"),
            )
            if clip_path:
                clips.append(clip_path)
        except Exception as e:
            logger.debug(f"Could not clip audio for result {i}: {e}")

    if not clips:
        logger.error("Failed to extract any audio clips.")
        print("Error: Could not extract audio for the results.")
        return

    # Stitch clips with beeps (same as personalization)
    from pydub.generators import Sine

    beep = Sine(config.PERS_BEEP_FREQ_HZ).to_audio_segment(
        duration=config.PERS_BEEP_DURATION_MS
    ).apply_gain(config.PERS_BEEP_GAIN_DB)

    combined = AudioSegment.empty()
    segment_lengths_ms = []
    for i, clip_path in enumerate(clips):
        seg = AudioSegment.from_mp3(clip_path)
        segment_lengths_ms.append(len(seg))
        if i > 0:
            combined += beep
        combined += seg

    # Save stitched MP3
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    output_audio = results_dir / f"{base_name}.mp3"
    combined.export(str(output_audio), format="mp3")
    logger.info(f"Stitched audio saved to: {output_audio}")

    # Write report (same style as personalization)
    output_text = results_dir / f"{base_name}.txt"
    ids = [r.get("id") for r in results[: len(clips)]]
    elapsed_ms = 0
    with open(output_text, "w", encoding="utf-8") as f:
        f.write("=== SEARCH RESULTS (TOP K STITCHED) ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Query: {query}\n")
        f.write(f"Top k: {len(clips)}\n")
        f.write(f"Story IDs (order): {ids}\n\n")

        for i, result in enumerate(results[: len(clips)]):
            story_id = result.get("id")
            summary = result.get("summary", "No summary")
            distance = result.get("distance", 0.0)
            transcript = result.get("transcript", "")
            marker = "START" if i == 0 else f"after {elapsed_ms / 1000.0:.2f}s (beep)"
            f.write(f"--- STORY {i + 1} (ID: {story_id} | Position: {marker}) ---\n")
            f.write(f"Similarity: {1 - distance:.4f}\n")
            f.write(f"SUMMARY: {summary}\n")
            # transcript is plain story text (new) or legacy JSON with "text" key
            try:
                story_data = json.loads(transcript)
                full_text = story_data.get("text", transcript)
            except Exception:
                full_text = transcript
            f.write(f"TEXT: {full_text.strip()}\n\n")

            elapsed_ms += segment_lengths_ms[i]
            if i < len(clips) - 1:
                elapsed_ms += config.PERS_BEEP_DURATION_MS

    logger.info(f"Report saved to: {output_text}")
    print(f"\nFound {len(clips)} results. Stitched audio and report written.")
    print(f"Audio: {output_audio}")
    print(f"Report: {output_text}")


def main() -> None:
    """Main entry point for the search CLI. Usage: python search.py <top_k> <query>"""
    if len(sys.argv) < 3:
        print("Usage: python search.py <top_k> <query>")
        print("\nExample:")
        print("  python search.py 5 climate change policy")
        sys.exit(1)

    try:
        top_k = int(sys.argv[1])
    except ValueError:
        print("Error: First argument must be a number (top k).")
        sys.exit(1)

    if top_k < 1:
        print("Error: top_k must be at least 1.")
        sys.exit(1)

    query = " ".join(sys.argv[2:])

    try:
        search_stories(query, top_k=top_k)
    except KeyboardInterrupt:
        logger.info("Search interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        print("Error: An unexpected error occurred. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
