"""
Search module for the radio archiver system.

This module provides a command-line interface for searching stored radio
stories using semantic similarity. It converts user queries into embeddings
and finds the most similar stories in the database.

Usage:
    python search.py <query text>
    
Example:
    python search.py climate change policy
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import database
import config
import json
import re
from pydub import AudioSegment
from utils import get_embedding, setup_logging


logger = setup_logging(__name__)


# Clipping constants
DEFAULT_TOP_K = config.SEARCH_DEFAULT_TOP_K
EXCERPT_LENGTH = config.SEARCH_EXCERPT_LENGTH
CLIP_PADDING_MS = config.CLIP_PADDING_MS


def extract_audio_segment(story_id: int, query: str = "clip", start_samples: Optional[int] = None, end_samples: Optional[int] = None, session_id: Optional[str] = None) -> Optional[str]:
    """
    Finds relevant MP3 chunks, concatenates them, and clips to the exact story range.
    Uses sample offsets and an optional session_id for 100% timing accuracy.
    """
    try:
        # Sanitize query for filename
        safe_query = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
        
        if start_samples is None or end_samples is None:
            logger.error("start_samples and end_samples are required for clipping")
            return None
            
        padding_ms = CLIP_PADDING_MS
        
        # 2. Find relevant files
        audio_files = sorted(list(config.AUDIO_DIR.glob("kqed_*.mp3")), reverse=True) # Check newest first
        if not audio_files:
            logger.warning("No audio files found for clipping")
            return None
            
        relevant_files = []
        
        # Use provided session_id or assume the latest one found in the files
        target_session = session_id
        if target_session and not target_session.startswith("kqed_"):
            target_session = "kqed_" + target_session
        
        for i, file_path in enumerate(audio_files):
            # Parse format: kqed_YYYYMMDD_HHMMSS[_ffffff]_OFFSET.mp3
            # Supports both legacy and new high-precision session IDs
            match = re.search(r'(kqed_\d{8}_\d{6}(?:_\d{6})?)_(\d{12})', file_path.stem)
            if not match: continue
                
            session_str = match.group(1)
            offset_samples = int(match.group(2))
            
            if target_session and session_str != target_session:
                continue # Only stick to the specified session
                
            # Sample-based check
            padding_samples = int((padding_ms / 1000.0) * config.VAD_SAMPLING_RATE)
            s_start = start_samples - padding_samples
            s_end = end_samples + padding_samples
            
            f_start = offset_samples
            f_end = offset_samples + config.SAMPLES_PER_ARCHIVE
            
            if f_start < s_end and f_end > s_start:
                if not target_session: target_session = session_str
                relevant_files.append((offset_samples, file_path))
        
        if not relevant_files:
            logger.warning(f"No audio files found for range samples=[{start_samples}, {end_samples}] session_id={session_id}")
            return None
            
        # Sort back to chronological order
        relevant_files.sort(key=lambda x: x[0])
            
        # 3. Concatenate relevant files
        combined = AudioSegment.empty()
        for _, file_path in relevant_files:
            segment = AudioSegment.from_mp3(str(file_path))
            combined += segment
            
        # 4. Calculate final clip offsets using GROUND TRUTH samples
        first_file_sample_offset = relevant_files[0][0]
        padding_samples = int((padding_ms / 1000.0) * config.VAD_SAMPLING_RATE)
        
        final_start_samples = (start_samples - padding_samples) - first_file_sample_offset
        final_end_samples = (end_samples + padding_samples) - first_file_sample_offset
        
        final_start = max(0, int((final_start_samples / config.VAD_SAMPLING_RATE) * 1000))
        final_end = min(len(combined), int((final_end_samples / config.VAD_SAMPLING_RATE) * 1000))
        
        # Ensure we have a valid clip (at least 100ms)
        if final_end <= final_start:
            logger.error(f"Invalid clip range: start={final_start}ms, end={final_end}ms")
            return None
        
        if final_end - final_start < 100:  # Minimum 100ms for valid MP3
            logger.warning(f"Clip too short ({final_end - final_start}ms), expanding to 100ms")
            final_end = final_start + 100
            if final_end > len(combined):
                final_start = max(0, len(combined) - 100)
                final_end = len(combined)
        
        clipped = combined[final_start:final_end]
        
        # 5. Save result with proper MP3 encoding
        results_dir = config.RESULTS_DIR
        results_dir.mkdir(exist_ok=True)
        
        output_path = results_dir / f"{safe_query}.mp3"
        
        # Export with explicit parameters to ensure complete, valid MP3 files
        # Use bitrate and parameters that ensure proper MP3 encoding
        clipped.export(
            str(output_path),
            format="mp3",
            bitrate="128k",
            parameters=["-q:a", "2"]  # High quality encoding
        )
        
        # Verify the file was created and has content
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
        
        # Create excerpt
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
    Search for stories matching the query.
    
    Args:
        query: Search query text
        top_k: Number of results to return
    """
    logger.info(f"Searching for: '{query}'")
    
    # Generate embedding for query
    embedding = get_embedding(query, task_type="RETRIEVAL_QUERY")
    
    if embedding is None:
        logger.error("Failed to generate embedding for query")
        print("Error: Could not process your query. Please try again.")
        return
    
    # Search database
    results = database.search_stories(embedding, top_k=top_k)
    
    # Display results
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        summary = result.get("summary", "No summary")
        distance = result.get("distance", 0.0)
        transcript = result.get("transcript", "")
        audio_path = result.get("audio_path", "{}")
        story_id = result.get("id")
        
        # Load and clip audio for ALL results
        audio_saved_at = None
        try:
            audio_meta = json.loads(audio_path)
            # Clip the audio for this story
            print(f"[{i}] Clipping audio for: {summary[:50]}...")
            audio_saved_at = extract_audio_segment(
                story_id,
                query=f"{query[:10]}_result{i}_s{story_id}",
                start_samples=audio_meta.get("start_samples"),
                end_samples=audio_meta.get("end_samples"),
                session_id=audio_meta.get("session_id")
            )
        except Exception as e:
            logger.debug(f"Could not clip audio for result {i}: {e}")
        
        # Parse full text
        try:
            # Transcript might be a JSON if it was stored as a blob
            story_data = json.loads(transcript)
            full_text = story_data.get("text", transcript)
        except:
            full_text = transcript
        
        print(f"{i}. {summary}")
        print(f"   Similarity: {1 - distance:.4f}")
        if audio_saved_at:
            print(f"   Audio Clip: {audio_saved_at}")
        print(f"   Transcript: {full_text.strip()}")
        print("-" * 60)


def main() -> None:
    """Main entry point for the search CLI."""
    if len(sys.argv) < 2:
        print("Usage: python search.py <query>")
        print("\nExample:")
        print("  python search.py climate change policy")
        sys.exit(1)
    
    # Join all arguments as the query
    query = " ".join(sys.argv[1:])
    
    try:
        search_stories(query)
    except KeyboardInterrupt:
        logger.info("Search interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        print(f"Error: An unexpected error occurred. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
