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
from utils import setup_logging, get_batch_embeddings, _build_embed_chunks


logger = setup_logging(__name__)


# Processing constants
MIN_STORY_LENGTH = config.MIN_STORY_LENGTH
MONITOR_INTERVAL = config.MONITOR_INTERVAL_S
GEMINI_MODEL = config.GEMINI_STORY_MODEL

TRANSCRIPT_FILE = config.TRANSCRIPT_DIR / "current_transcript.txt"
STORIES_LOG_FILE = config.STORY_DIR / "stories_log.jsonl"
LLM_STORIES_BATCH_FILE = config.STORY_DIR / "llm_stories_batch.jsonl"  # LLM-returned stories persisted before embedding/DB
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
    Below is a transcript of a continuous live radio feed. 
    
    Your goal is identifying complete stories that stand alone as a story.
    All consecutive information should be included in a single story. Include things like speaker introductions and outros.

    CRITICAL INSTRUCTIONS:
    1. Every line should be included in a story. Return a list of separate story objects.
    2. For each complete story, provide:
        - "summary": A concise title or 1-sentence summary of the story/topic. Should be isolated enough to not need "and".
        - "category": Categorize the story into "major news", "highlight", "think piece", "journey of discovery", or "filler".
        - "start_index": The index [i] of the first sentence of the story.
        - "end_index": The index [i] of the last sentence of the story.
    3. Ensure the output is a valid JSON array of objects.
    
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


def save_llm_stories_batch(
    window_start: int,
    window_end: int,
    stories: List[Dict[str, Any]],
) -> None:
    """
    Persist LLM-returned stories to a text file so they are not in limbo.
    Called right after we get valid_stories; transcript line counter is incremented after this.
    Each line is JSON: {"window_start": int, "window_end": int, "stories": [...]}.
    """
    try:
        config.STORY_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "window_start": window_start,
            "window_end": window_end,
            "stories": stories,
        }
        with open(LLM_STORIES_BATCH_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Stored {len(stories)} LLM stories to {LLM_STORIES_BATCH_FILE.name} (window {window_start}->{window_end})")
    except IOError as e:
        logger.error(f"Error saving LLM stories batch to file: {e}")
        raise


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
    
    new_index = current_index + (config.WINDOW_SIZE_SENTENCES - config.OVERLAP_SENTENCES)
    save_llm_stories_batch(current_index, new_index, valid_stories)
    save_processed_index(new_index)
    logger.info(f"Advanced transcript line index to {new_index}")
    
    # Build story blobs and chunk by char limit (same as embedding API)
    story_blobs = []
    for story in valid_stories:
        for field in ["text", "summary", "adjectives"]:
            if isinstance(story.get(field), list):
                story[field] = " ".join([str(x) for x in story[field]])
        story_blobs.append(json.dumps(story, ensure_ascii=False))
    
    chunks_blobs = _build_embed_chunks(story_blobs, config.EMBED_BATCH_MAX_CHARS)
    total_chunks = len(chunks_blobs)
    start_idx = 0
    total_saved = 0

    for chunk_num, chunk_blobs in enumerate(chunks_blobs, start=1):
        chunk_stories = valid_stories[start_idx : start_idx + len(chunk_blobs)]
        start_idx += len(chunk_blobs)

        logger.info(f"Embedding chunk {chunk_num}/{total_chunks} ({len(chunk_blobs)} stories)...")
        chunk_embeddings = get_batch_embeddings(chunk_blobs)
        if not chunk_embeddings or len(chunk_embeddings) != len(chunk_stories):
            logger.error("Failed to generate batch embeddings for chunk %s", chunk_num)
            return False

        # Save this chunk to DB immediately.
        # transcript = same text we embedded (may be truncated to EMBED_BATCH_MAX_CHARS for long stories).
        # Audio refs (start_time, end_time, audio_path) are always the full story segment for search/personalization.
        for idx, story in enumerate(chunk_stories):
            story_blob = chunk_blobs[idx]
            embedding = chunk_embeddings[idx]
            matching_segments = story.get("matching_segments", [])

            if matching_segments:
                start_time = min(s["start"] for s in matching_segments)
                end_time = max(s["end"] for s in matching_segments)
            else:
                start_time = window_data[0]["start"]
                end_time = window_data[-1]["end"]

            log_story(story)
            if embedding is None:
                logger.warning("Failed to generate embedding for story, skipping storage")
                continue

            try:
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
                        session_dt = datetime.now().replace(microsecond=0)
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
            except Exception as e:
                logger.warning("Failed to create detailed audio metadata: %s", e)
                audio_info = f"sliding_window_log:{STORIES_LOG_FILE.name}"
                db_timestamp = datetime.now().isoformat()

            story_id = database.save_story(
                timestamp=db_timestamp,
                start_time=start_time,
                end_time=end_time,
                transcript=story_blob,
                summary=story.get("summary", ""),
                audio_path=audio_info,
                embedding=embedding
            )
            if story_id is None:
                logger.error("Failed to save story to database")
            else:
                total_saved += 1

        if chunk_num < total_chunks:
            logger.info("Chunk %s/%s saved; sleeping %s s before next embedding chunk", chunk_num, total_chunks, config.EMBED_BATCH_SLEEP_SECONDS)
            time.sleep(config.EMBED_BATCH_SLEEP_SECONDS)

    logger.info("Saved %s stories to database", total_saved)
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
            # print(f"running without processing stories")
            # processed = False
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
