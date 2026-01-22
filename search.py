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


def extract_audio_segment(start_iso: str, end_iso: str, story_id: int, query: str = "clip") -> Optional[str]:
    """
    Finds relevant MP3 chunks, concatenates them, and clips to the exact story range.
    """
    try:
        # Sanitize query for filename
        safe_query = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')[:30]
        
        # 1. Expand range based on requested padding
        base_start_dt = datetime.fromisoformat(start_iso)
        base_end_dt = datetime.fromisoformat(end_iso)
        
        start_dt = base_start_dt - timedelta(milliseconds=CLIP_PADDING_MS)
        end_dt = base_end_dt + timedelta(milliseconds=CLIP_PADDING_MS)
        
        # 2. Find relevant files covering the EXPANDED range
        audio_files = sorted(list(config.AUDIO_DIR.glob("kqed_*.mp3")))
        if not audio_files:
            logger.warning("No audio files found for clipping")
            return None
            
        relevant_files = []
        for i, file_path in enumerate(audio_files):
            # Parse timestamp from filename: kqed_20260120_050000.mp3
            ts_str = file_path.stem.split("_", 1)[1]
            file_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            
            # File is relevant if its lifetime overlaps with our [start_dt, end_dt] window
            file_end_dt = file_dt + timedelta(seconds=config.CHUNK_DURATION + 1) # +1s wiggle for safety
            
            if file_dt < end_dt and file_end_dt > start_dt:
                relevant_files.append((file_dt, file_path))
        
        if not relevant_files:
            logger.warning(f"No audio files found covering expanded range {start_dt} to {end_dt}")
            return None
            
        # 3. Concatenate relevant files
        combined = AudioSegment.empty()
        first_file_dt = relevant_files[0][0]
        
        for _, file_path in relevant_files:
            segment = AudioSegment.from_mp3(str(file_path))
            combined += segment
            
        # 4. Calculate final clip offsets relative to the concatenated start
        start_offset_ms = int((start_dt - first_file_dt).total_seconds() * 1000)
        duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
        
        # Clamp to available audio bounds
        final_start = max(0, start_offset_ms)
        final_end = min(len(combined), start_offset_ms + duration_ms)
        
        clipped = combined[final_start:final_end]
        
        # 4. Save result
        results_dir = config.DATA_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        
        output_path = results_dir / f"{safe_query}.mp3"
        clipped.export(str(output_path), format="mp3")
        
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
            files = audio_metadata.get("files", [])
            start_ts = audio_metadata.get("start_timestamp", "Unknown")
            duration = audio_metadata.get("duration_seconds", 0)
            
            # Format timestamp for display
            if start_ts != "Unknown":
                try:
                    dt = datetime.fromisoformat(start_ts)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = start_ts
            else:
                time_str = "Unknown"
            
            audio_info = f"{time_str} ({duration}s, {len(files)} file{'s' if len(files) != 1 else ''})"
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
            if "start_time" in audio_meta and "end_time" in audio_meta:
                # Clip the audio for this story
                print(f"[{i}] Clipping audio for: {summary[:50]}...")
                audio_saved_at = extract_audio_segment(
                    audio_meta["start_time"], 
                    audio_meta["end_time"], 
                    story_id,
                    query=f"{query}_result{i}_story{story_id}"
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
