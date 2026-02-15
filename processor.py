"""
Story processing and segmentation module for the radio archiver system.

This module monitors transcript files, chunks them by character count (whole
lines only, up to LLM_CONTEXT_MAX_CHARS), and uses Gemma 3 27B to segment
into stories. Embeddings use Gemini (unchanged). Stories are stored in the DB.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime, timedelta

from google import genai

import config
import database
from utils import setup_logging, get_batch_embeddings, _build_embed_chunks


logger = setup_logging(__name__)


# Processing constants
MIN_STORY_LENGTH = config.MIN_STORY_LENGTH
MONITOR_INTERVAL = config.MONITOR_INTERVAL_S
LLM_MODEL = config.LLM_STORY_MODEL
LLM_CONTEXT_MAX_CHARS = config.LLM_CONTEXT_MAX_CHARS
LLM_OVERLAP_MAX_CHARS = config.LLM_OVERLAP_MAX_CHARS

TRANSCRIPT_FILE = config.TRANSCRIPT_DIR / "current_transcript.txt"
STORIES_LOG_FILE = config.STORY_DIR / "stories_log.jsonl"
SIMPLIFIED_STORIES_LOG_FILE = config.STORY_DIR / "simplified_stories_log.jsonl"
LLM_STORIES_BATCH_FILE = config.STORY_DIR / "llm_stories_batch.jsonl"  # LLM-returned stories persisted before embedding/DB
STATE_FILE = config.DATA_DIR / "processing_state.json"


def segment_transcripts(combined_text: str) -> Optional[List[Dict[str, str]]]:
    """
    Use Gemma 3 27B to segment combined transcript into distinct stories.
    """
    prompt = f"""You are an editor for a radio station.
Below is a transcript of a continuous live radio feed.

Your goal is identifying complete stories on a single topic.
All consecutive information should be included in a single story. Include things like speaker introductions and outros.

This is only a section of the transcript, so if the final lines are part of a story that has not ended, then do not include these lines as part of a story.
The marker "NO SPEECH DETECTED FOR X SEC" indicates music or other audio and can be part of a story.
Advertisements and filler should be separated into their own stories so they don't interfere with actual stories.

CRITICAL INSTRUCTIONS:
1. Every line should be included as part of a story, except for stories that have not ended yet. Return a list of separate story objects.
2. Each story must be exactly one topic. Do not combine unrelated items into one story. The summary must describe all content in that story's segment.
3. For each complete story, provide:
   - "summary": A concise title or 1-sentence summary covering the entire story.
   - "category": Categorize the story into "major news", "highlight", "think piece", "journey of discovery", or "filler".
   - "start_index": The 0-based index of the first sentence.
   - "end_index": The 0-based index of the last sentence (inclusive)
4. Output only a valid JSON array of objects, no other text.

Transcript sentences:
{combined_text}
"""

    try:
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
        )
        raw = (response.text or "").strip()
        # Strip markdown code fence if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        stories = json.loads(raw)
        logger.info(f"Segmented transcript into {len(stories)} stories")
        return stories
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error segmenting transcripts with LLM: {e}")
        return None


def _char_count(sentences: List[Dict[str, Any]]) -> int:
    """Length of the string sent to the LLM: \\n.join([f'[{i}] {s[text]}' ...])."""
    return sum(len(f"[{i}] {s['text']}") + 1 for i, s in enumerate(sentences))


def _tail_up_to_chars(
    sentences: List[Dict[str, Any]], max_chars: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Return the suffix of sentences that fits in max_chars (whole lines only).
    Returns (tail_list, char_count). Avoids recounting: caller can store char_count.
    """
    if not sentences or max_chars <= 0:
        return [], 0
    tail: List[Dict[str, Any]] = []
    total = 0
    for j in range(len(sentences) - 1, -1, -1):
        # As sent to LLM, this line is "[i] text\n" with i = len(tail) when at start of tail
        line_len = len(f"[{len(tail)}] {sentences[j]['text']}") + 1
        if total + line_len > max_chars and tail:
            break
        tail.insert(0, sentences[j])
        total += line_len
    return tail, total


def get_processing_state() -> Dict[str, Any]:
    """Read full processing state (last_index, overlap_start_index, overlap_char_count)."""
    default = {
        "last_index": 0,
        "overlap_start_index": 0,
        "overlap_char_count": 0,
    }
    if not STATE_FILE.exists():
        return default
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
            # Migrate old state that stored overlap_sentences
            if "overlap_sentences" in state and "overlap_start_index" not in state:
                overlap = state.get("overlap_sentences") or []
                state["overlap_start_index"] = state["last_index"] - len(overlap) if overlap else state["last_index"]
                del state["overlap_sentences"]
            default.update(state)
            return default
    except (json.JSONDecodeError, IOError):
        return default


def save_processing_state(
    last_index: int,
    overlap_start_index: Optional[int] = None,
    overlap_char_count: int = 0,
) -> None:
    """Save processing state. Overlap is stored as start index only; sentences are re-read from transcript."""
    try:
        if overlap_start_index is None:
            overlap_start_index = last_index  # no overlap
        state = {
            "last_index": last_index,
            "overlap_start_index": overlap_start_index,
            "overlap_char_count": overlap_char_count,
            "updated_at": time.time(),
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Failed to save state: {e}")


def get_last_processed_index() -> int:
    """Read the last processed line index from state file."""
    return get_processing_state()["last_index"]


def save_processed_index(index: int) -> None:
    """Save only last_index (keeps existing overlap_start_index and overlap_char_count if present)."""
    state = get_processing_state()
    state["last_index"] = index
    state["updated_at"] = time.time()
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
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


def get_sentences_from_file_up_to_chars(
    start_index: int,
    max_chars: int,
    initial_sentences: Optional[List[Dict[str, Any]]] = None,
    initial_char_count: int = 0,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    """
    Build one chunk for the LLM: optional overlap first, then new lines from file.
    Never splits a line. Uses initial_char_count so we don't recount the overlap.

    Character count is the same as the string sent to the LLM:
    "\\n".join([f"[{i}] {s['text']}" for i, s in enumerate(sentences)]).

    Returns:
        (full list of sentence dicts, next_index, hit_char_limit).
        hit_char_limit is True when we stopped because the next line would exceed max_chars.
    """
    sentences: List[Dict[str, Any]] = list(initial_sentences) if initial_sentences else []
    total_chars = initial_char_count
    num_new_added = 0
    hit_char_limit = False

    if not TRANSCRIPT_FILE.exists() or max_chars <= 0:
        next_index = start_index if not initial_sentences else start_index
        return sentences, next_index, hit_char_limit

    sentence_counter = 0
    try:
        with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if sentence_counter < start_index:
                    sentence_counter += 1
                    continue

                parts = line.split("|")
                if len(parts) < 4:
                    continue

                ts = parts[0]
                if not re.fullmatch(r"\d{6}_\d{6}", ts):
                    continue

                try:
                    start_s = int(parts[1])
                    end_s = int(parts[2])
                    text = "|".join(parts[3:])
                except ValueError:
                    continue

                expanded = "20" + ts
                session_id = "kqed_" + expanded
                seg = {
                    "session_id": session_id,
                    "start_samples": start_s,
                    "end_samples": end_s,
                    "start": start_s / config.VAD_SAMPLING_RATE,
                    "end": end_s / config.VAD_SAMPLING_RATE,
                    "text": text,
                    "timestamp": expanded,
                }

                line_repr_len = len(f"[{len(sentences)}] {text}") + 1
                if total_chars + line_repr_len > max_chars and sentences:
                    hit_char_limit = True
                    break

                sentences.append(seg)
                total_chars += line_repr_len
                num_new_added += 1
                sentence_counter += 1
    except IOError as e:
        logger.error(f"Error reading transcript file: {e}")

    next_index = start_index + num_new_added
    return sentences, next_index, hit_char_limit


def log_story(story_json: Dict[str, Any]) -> None:
    """
    Append a story JSON object to the stories log file with pretty formatting.
    Also appends a simplified version (summary, category, start_index, end_index, text, word_count only)
    to simplified_stories_log.jsonl. Both files include word_count and use one entry per block (separate lines).
    
    Args:
        story_json: Story data as a dictionary
    """
    try:
        config.STORY_DIR.mkdir(parents=True, exist_ok=True)
        text = story_json.get("text", "")
        word_count = len(text.split())
        full_entry = {**story_json, "word_count": word_count}
        with open(STORIES_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(full_entry, ensure_ascii=False, indent=4) + "\n---\n")
        simplified = {
            "summary": story_json.get("summary", ""),
            "category": story_json.get("category", ""),
            "start_index": story_json.get("start_index"),
            "end_index": story_json.get("end_index"),
            "text": text,
            "word_count": word_count,
        }
        with open(SIMPLIFIED_STORIES_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(simplified, ensure_ascii=False, indent=4) + "\n")
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
    Process one chunk (up to LLM_CONTEXT_MAX_CHARS) using Gemma. Chunk starts with
    up to LLM_OVERLAP_MAX_CHARS from the previous chunk (whole lines), then new
    lines from file. State stores overlap so we don't recount.
    """
    state = get_processing_state()
    current_index = state["last_index"]
    overlap_start = state.get("overlap_start_index", current_index)
    overlap_char_count = state.get("overlap_char_count") or 0
    # Re-read overlap sentences from transcript when we have a valid overlap range
    if overlap_start < current_index and overlap_char_count > 0:
        overlap_sentences = get_sentences_from_file(current_index - overlap_start, overlap_start)
        if len(overlap_sentences) != current_index - overlap_start:
            overlap_sentences = []
            overlap_char_count = 0
    else:
        overlap_sentences = []
        overlap_char_count = 0

    window_data, next_index, hit_char_limit = get_sentences_from_file_up_to_chars(
        current_index,
        LLM_CONTEXT_MAX_CHARS,
        initial_sentences=overlap_sentences if overlap_sentences else None,
        initial_char_count=overlap_char_count,
    )

    if not window_data:
        return False

    # Only send when chunk is "full" (we stopped because next line would exceed limit)
    if not hit_char_limit:
        return False

    total_chars = _char_count(window_data)

    num_overlap = len(overlap_sentences)
    if num_overlap:
        print(
            f"Sending {len(window_data)} lines to Gemma (overlap: first {num_overlap} lines, new: {current_index} to {next_index - 1}), ~{total_chars} chars"
        )
        logger.info(
            "Sending %s lines to Gemma (overlap: first %s, new: %s to %s), ~%s chars",
            len(window_data), num_overlap, current_index, next_index - 1, total_chars,
        )
    else:
        line_end = next_index - 1
        print(f"Sending lines {current_index} to {line_end} ({len(window_data)} lines) to Gemma, ~{total_chars} chars")
        logger.info(
            "Sending lines %s to %s (%s lines) to Gemma, ~%s chars",
            current_index, line_end, len(window_data), total_chars,
        )

    combined_text = "\n".join([f"[{i}] {s['text']}" for i, s in enumerate(window_data)])
    stories = segment_transcripts(combined_text)
    
    if stories is None:
        logger.error("Story segmentation failed (API error or JSON parse error)")
        return False

    logger.info(f"LLM returned {len(stories)} raw stories")
    
    # Function to get text from indexed range (LLM uses 0-based indices from [i] in the transcript)
    def reconstruct_story_data(s):
        try:
            start_idx = int(s.get("start_index", -1))
            end_idx = int(s.get("end_index", -1))
            if start_idx == -1 or end_idx == -1:
                return None
            # Some models return 1-based indices; if so, indices are in 1..len(window_data)
            n = len(window_data)
            if start_idx >= 1 and end_idx == n and n > 0:
                start_idx, end_idx = start_idx - 1, end_idx - 1
            elif start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            # Clamp to valid range
            start_idx = max(0, min(start_idx, n - 1))
            end_idx = max(start_idx, min(end_idx, n - 1))
            matching = window_data[start_idx : end_idx + 1]
            s["text"] = " ".join([m["text"] for m in matching])
            s["matching_segments"] = matching  # used for timing/audio below
            return s
        except (ValueError, TypeError):
            return None

    # Reconstruct exact text and filter
    valid_stories = []
    for s in stories:
        s = reconstruct_story_data(s)
        if s and len(s["text"]) >= MIN_STORY_LENGTH:
            valid_stories.append(s)
    
    overlap_tail, overlap_tail_count = _tail_up_to_chars(window_data, LLM_OVERLAP_MAX_CHARS)

    overlap_start_index = next_index - len(overlap_tail)

    if not valid_stories:
        logger.warning(f"No valid stories after filtering (Raw stories: {len(stories)})")
        save_processing_state(next_index, overlap_start_index, overlap_tail_count)
        return True

    logger.info(f"Found {len(valid_stories)} valid stories")
    save_llm_stories_batch(current_index, next_index, valid_stories)
    save_processing_state(next_index, overlap_start_index, overlap_tail_count)
    logger.info(f"Advanced transcript line index to {next_index}")
    
    # Normalize list fields for logging/DB summary; then embed only the actual story text
    for story in valid_stories:
        for field in ["text", "summary", "adjectives"]:
            if isinstance(story.get(field), list):
                story[field] = " ".join([str(x) for x in story[field]])
    story_texts = [story.get("text", "") for story in valid_stories]
    # Chunk by character limit (plain text only); transcript column will store this same text
    chunks_blobs = _build_embed_chunks(story_texts, config.EMBED_BATCH_MAX_CHARS)
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
        # transcript = plain story text we embedded (same as chunk_blobs; may be truncated for long stories).
        # summary = from story dict (DB column); search/personalization read summary from DB, not from transcript.
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
    Monitor transcript file and process in character-based chunks.
    Reads up to LLM_CONTEXT_MAX_CHARS per chunk (whole lines only), segments
    with Gemma, then advances the line index in processing_state.json.
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
