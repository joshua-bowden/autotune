"""
Story processing and segmentation module for the radio archiver system.

This module monitors transcript files, batches them together, and uses
Google's Gemini AI to:
1. Segment continuous radio transcripts into distinct stories
2. Generate summaries for each story
3. Create embeddings for semantic search
4. Store processed stories in the database

The module implements rate limiting to stay within Gemini API free tier limits.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime, timedelta

from google import genai
from google.genai import types

import config
import database
from utils import setup_logging, get_batch_embeddings


logger = setup_logging(__name__)


# Processing constants
MIN_STORY_LENGTH = config.MIN_STORY_LENGTH
MONITOR_INTERVAL = config.MONITOR_INTERVAL_S
GEMINI_MODEL = config.GEMINI_STORY_MODEL

TRANSCRIPT_FILE = config.TRANSCRIPT_DIR / "current_transcript.txt"
STORIES_LOG_FILE = config.STORY_DIR / "stories_log.jsonl"
STATE_FILE = config.DATA_DIR / "processing_state.json"


def segment_transcripts(combined_text: str) -> Optional[List[Dict[str, str]]]:
    """
    Use Gemini to segment combined transcript into distinct stories.
    
    Args:
        combined_text: Combined transcript text from multiple files
        duration_minutes: Approximate duration of audio in minutes
        
    Returns:
        List of story dictionaries with 'summary' and 'text' keys,
        or None if segmentation fails
    """
    prompt = f"""
    You are an editor for a radio station.
    Below is a transcript of a continuous live radio feed, provided as a numbered list of sentence fragments. 
    
    Your goal is identifying sharp, complete stories that can stand alone as a short radio highlight clip.
    A proper story is a hook that introduces the topic, talks briefly about it, and leaves the listener wanting to hear more.
    
    While there are likely transcription errors, make sure that your selected story starts with good grammar
    at the beginning of a sentence and ends at the end of a sentence.

    CRITICAL INSTRUCTIONS:
    1. If the transcript contains multiple stories that satisfy this criteria, split them into separate story objects.
    2. For each complete story, provide:
        - "summary": A concise title or 1-sentence summary of the story/topic. Should be isolated enough to not need "and".
        - "adjectives": 5 adjectives to describe the how a person would feel after hearing this story.
        - "start_index": The index [i] of the first sentence of the story.
        - "end_index": The index [i] of the last sentence of the story.
    3. Ignore filler sentences, ads, or station IDs between stories.
    4. Ensure the output is a valid JSON array of objects.
    
    Transcript sentences:
    {combined_text}
    """
    
    try:
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        stories = json.loads(response.text)
        logger.info(f"Segmented transcript into {len(stories)} stories")
        return stories
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error segmenting transcripts with Gemini: {e}")
        return None


def get_last_processed_index() -> int:
    """Read the last processed line index from state file."""
    if not STATE_FILE.exists():
        return 0
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            return state.get("last_index", 0)
    except (json.JSONDecodeError, IOError):
        return 0

def save_processed_index(index: int):
    """Save the last processed line index to state file."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"last_index": index, "updated_at": time.time()}, f)
    except IOError as e:
        logger.error(f"Failed to save state: {e}")

def get_sentences_from_file(count: int, start_index: int) -> List[Dict[str, Any]]:
    """
    Read N sentences from the transcript file starting from a specific index.
    Parses transcript lines.
    Supported formats:
    - YYMMDD_HHMMSS|START_SAMPLES|END_SAMPLES|text
    
    Args:
        count: Number of sentences to read
        start_index: Sentence index to start reading from (0-indexed, excludes markers)
        
    Returns:
        List of dictionaries with keys: session_id, start_samples, end_samples, start, end, text
    """
    if not TRANSCRIPT_FILE.exists():
        return []
        
    sentences = []
    sentence_counter = 0
    
    try:
        with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Sentence line
                if sentence_counter < start_index:
                    sentence_counter += 1
                    continue

                parts = line.split("|")
                if len(parts) < 4:
                    continue

                ts = parts[0]
                if not re.fullmatch(r"\d{6}_\d{6}", ts):
                    # No backwards compatibility: skip non-conforming lines
                    continue

                try:
                    start_s = int(parts[1])
                    end_s = int(parts[2])
                    text = "|".join(parts[3:])
                except ValueError:
                    continue

                # Expand YY -> YYYY for session matching & ISO computation
                expanded = "20" + ts  # 20YYMMDD_HHMMSS
                session_id = "kqed_" + expanded

                sentences.append({
                    "session_id": session_id,
                    "start_samples": start_s,
                    "end_samples": end_s,
                    "start": start_s / config.VAD_SAMPLING_RATE,
                    "end": end_s / config.VAD_SAMPLING_RATE,
                    "text": text,
                    "timestamp": expanded,
                })
                sentence_counter += 1
                
                if len(sentences) >= count:
                    break
    except IOError as e:
        logger.error(f"Error reading transcript file: {e}")
        
    return sentences


def log_story(story_json: Dict[str, Any]) -> None:
    """
    Append a story JSON object to the stories log file with pretty formatting.
    
    Args:
        story_json: Story data as a dictionary
    """
    try:
        with open(STORIES_LOG_FILE, "a", encoding="utf-8") as f:
            # Each key on a separate line (pretty print)
            f.write(json.dumps(story_json, ensure_ascii=False, indent=4) + "\n---\n")
    except IOError as e:
        logger.error(f"Error logging story to file: {e}")


def process_sentences() -> bool:
    """
    Process a window of sentences using Gemini.
    
    Reads window of N sentences from the current index, segments into stories, 
    and increments index by (window_size - overlap).
    
    Returns:
        True if processing occurred, False otherwise
    """
    current_index = get_last_processed_index()
    window_data = get_sentences_from_file(config.WINDOW_SIZE_SENTENCES, current_index)
    
    if len(window_data) < config.WINDOW_SIZE_SENTENCES:
        # logger.info(f"Window not full yet ({len(window_data)}/{config.WINDOW_SIZE_SENTENCES} segments). Waiting...")
        return False
        
    logger.info(f"Window full! Processing {len(window_data)} segments starting at index {current_index}...")
    
    # Pass indices to Gemini for precise range tracking
    # We explicitly only send the 'text' field, which has timestamps already stripped during file read
    combined_text = "\n".join([f"[{i}] {s['text']}" for i, s in enumerate(window_data)])
    
    # Segment into stories using Gemini
    stories = segment_transcripts(combined_text)
    
    if stories is None:
        logger.error("Story segmentation failed (API error or JSON parse error)")
        return False
    
    logger.info(f"Gemini returned {len(stories)} raw stories")
    
    # Do not stall for rate limit; assume that it will take longer to fill the window
    # logger.info(f"Waiting {config.LLM_RATE_LIMIT_DELAY_S} seconds for LLM rate limit...")
    # time.sleep(config.LLM_RATE_LIMIT_DELAY_S)
    
    # Function to get text from indexed range
    def reconstruct_story_data(s):
        try:
            start_idx = int(s.get("start_index", -1))
            end_idx = int(s.get("end_index", -1))
            
            if start_idx == -1 or end_idx == -1:
                return None
            
            # Clamp and extract
            start_idx = max(0, min(start_idx, len(window_data) - 1))
            end_idx = max(start_idx, min(end_idx, len(window_data) - 1))
            
            matching = window_data[start_idx : end_idx + 1]
            s["text"] = " ".join([m["text"] for m in matching])
            s["matching_segments"] = matching # Temporary for timing logic below
            return s
        except (ValueError, TypeError):
            return None

    # Reconstruct exact text and filter
    valid_stories = []
    for s in stories:
        s = reconstruct_story_data(s)
        if s and len(s["text"]) >= MIN_STORY_LENGTH:
            valid_stories.append(s)
    
    if not valid_stories:
        logger.warning(f"No valid stories after filtering (Raw stories: {len(stories)})")
        
        # Still advance index so we don't get stuck
        new_index = current_index + (config.WINDOW_SIZE_SENTENCES - config.OVERLAP_SENTENCES)
        save_processed_index(new_index)
        return True
    
    logger.info(f"Found {len(valid_stories)} valid stories")
    
    # Collect all story blobs for batch embedding
    story_blobs = []
    for story in valid_stories:
        # Normalize fields
        for field in ["text", "summary", "adjectives"]:
            if isinstance(story.get(field), list):
                story[field] = " ".join([str(x) for x in story[field]])
        story_blobs.append(json.dumps(story, ensure_ascii=False))
    
    # Generate all embeddings in one batch call (efficient)
    logger.info(f"Generating batch embeddings for {len(story_blobs)} stories...")
    all_embeddings = get_batch_embeddings(story_blobs)
    
    if not all_embeddings or len(all_embeddings) != len(valid_stories):
        logger.error("Failed to generate batch embeddings")
        return False

    # Process each story and save to DB
    for idx, story in enumerate(valid_stories):
        story_blob = story_blobs[idx]
        embedding = all_embeddings[idx]
        story_text = story.get("text", "").strip()
        
        # Use the segments we retrieved via indices
        matching_segments = story.get("matching_segments", [])
        
        if matching_segments:
            start_time = min(s["start"] for s in matching_segments)
            end_time = max(s["end"] for s in matching_segments)
            base_timestamp = matching_segments[0]["timestamp"]
        else:
            # Should not happen with new logic
            start_time = window_data[0]["start"]
            end_time = window_data[-1]["end"]
            base_timestamp = window_data[0]["timestamp"]

        # Log to file with pretty formatting
        log_story(story)

        if embedding is None:
            logger.warning(f"Failed to generate embedding for story, skipping storage")
            continue
            
        # Prepare detailed metadata for audio clipping
        try:
            # Session ID (prefer per-line timestamp-derived session_id)
            session_id = matching_segments[0].get("session_id") or "unknown"
            start_samples = matching_segments[0]["start_samples"]
            end_samples = matching_segments[-1]["end_samples"]
            
            session_id_clean = session_id.replace("kqed_", "") if session_id.startswith("kqed_") else session_id
            try:
                session_dt = datetime.strptime(session_id_clean, "%Y%m%d_%H%M%S_%f")
            except ValueError:
                try:
                    session_dt = datetime.strptime(session_id_clean, "%Y%m%d_%H%M%S")
                except ValueError:
                    session_dt = datetime.now().replace(microsecond=0) # Last resort fallback
            
            start_iso = (session_dt + timedelta(seconds=start_samples / config.VAD_SAMPLING_RATE)).isoformat()
            end_iso = (session_dt + timedelta(seconds=end_samples / config.VAD_SAMPLING_RATE)).isoformat()
            
            audio_info = json.dumps({
                "start_time": start_iso,
                "end_time": end_iso,
                "start_samples": start_samples,
                "end_samples": end_samples,
                "session_id": session_id,
                "duration": (end_samples - start_samples) / config.VAD_SAMPLING_RATE
            })
            db_timestamp = start_iso
        except (Exception) as e:
            logger.warning(f"Failed to create detailed audio metadata: {e}")
            audio_info = f"sliding_window_log:{STORIES_LOG_FILE.name}"
            db_timestamp = datetime.now().isoformat()
            
        story_id = database.save_story(
            timestamp=db_timestamp,
            start_time=start_time,
            end_time=end_time,
            transcript=story_blob, # Store the full JSON chunk
            summary=story.get("summary", ""),
            audio_path=audio_info,
            embedding=embedding
        )
        
        if story_id is None:
            logger.error(f"Failed to save story to database")
        else:
            logger.info(f"Saved story to database(ID: {story_id})")
            
    # Update index (sliding window: shift by window_size - overlap)
    new_index = current_index + (config.WINDOW_SIZE_SENTENCES - config.OVERLAP_SENTENCES)
    save_processed_index(new_index)
    logger.info(f"Advanced index to {new_index}")
    
    return True


def segment_and_process_stories() -> None:
    """
    Monitor transcript file and process in sliding windows.
    
    Continuously watches current_transcript.txt, reads windows of sentences,
    and processes them using Gemini. Runs until interrupted.
    """
    logger.info(f"Monitoring {TRANSCRIPT_FILE} for sentences to process...")
    
    try:
        while True:
            # Process sentences if enough are available
            processed = process_sentences()
            
            # If nothing was processed, wait a bit
            if not processed:
                # logger.debug(f"Sleeping for {MONITOR_INTERVAL}s...")
                time.sleep(MONITOR_INTERVAL)
            else:
                # If we processed a window, immediately check for another
                # (to catch up if we have a backlog)
                pass
            
    except KeyboardInterrupt:
        logger.info("Story processing stopped")
    except Exception as e:
        logger.error(f"Unexpected error in processing loop: {e}")


if __name__ == "__main__":
    segment_and_process_stories()
