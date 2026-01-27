"""
Live transcription from radio stream with natural settlement and tracked period timestamps.
Uses Moonshine and SileroVAD ONNX models.
"""

import argparse
import os
import signal
import time
import subprocess
import threading
import requests
import logging
import numpy as np
import torch
from datetime import datetime, timedelta
from queue import Queue
from collections import deque
from silero_vad import VADIterator, load_silero_vad

import config
from utils import setup_logging, Transcriber, normalize_word

logger = setup_logging(__name__)

def write_to_transcript(start_samples, end_samples, text):
    """Appends to the transcript file in the specific format with only sample offsets."""
    os.makedirs(config.TRANSCRIPT_DIR, exist_ok=True)
    with open(config.TRANSCRIPT_DIR / "current_transcript.txt", "a", encoding="utf-8") as f:
        f.write(f"{start_samples}|{end_samples}|{text.strip()}\n")

def write_session_marker(session_start_dt):
    """Writes a session marker to the transcript to anchor sample offsets."""
    session_str = session_start_dt.strftime("%Y%m%d_%H%M%S_%f")
    os.makedirs(config.TRANSCRIPT_DIR, exist_ok=True)
    with open(config.TRANSCRIPT_DIR / "current_transcript.txt", "a", encoding="utf-8") as f:
        f.write(f"# SESSION {session_str}\n")

def run_ffmpeg(pcm_queue, shutdown_event):
    """Pipes raw stream bytes to ffmpeg and reads back PCM."""
    cmd = [
        "ffmpeg",
        "-i", "pipe:0",          # Input from stdin
        "-f", "s16le",          # Output format raw pcm 16-bit
        "-ar", str(config.VAD_SAMPLING_RATE),
        "-ac", "1",             # Mono
        "pipe:1"                # Output to stdout
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1024 * 1024
    )
    
    def read_pcm():
        pcm_buffer = bytearray()
        byte_chunk_size = config.VAD_CHUNK_SIZE * 2
        while not shutdown_event.is_set():
            try:
                chunk = proc.stdout.read(byte_chunk_size)
                if not chunk:
                    break
                pcm_buffer.extend(chunk)
                while len(pcm_buffer) >= byte_chunk_size:
                    to_process = pcm_buffer[:byte_chunk_size]
                    pcm_buffer = pcm_buffer[byte_chunk_size:]
                    # Put raw int16 bytes into the queue to keep it fast and compatible with encoder
                    pcm_queue.put(to_process)
            except Exception:
                break
        proc.terminate()

    threading.Thread(target=read_pcm, daemon=True).start()
    return proc

class LiveTranscriberEngine:
    def __init__(self, model_name):
        self.transcribe = Transcriber(model_name)
        self.vad_model = load_silero_vad(onnx=True)
        self.vad_iterator = VADIterator(
            model=self.vad_model,
            sampling_rate=config.VAD_SAMPLING_RATE,
            threshold=config.VAD_THRESHOLD,
            min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
        )
        
        self.pcm_queue = Queue()
        self.shutdown_event = threading.Event()
        self.ffmpeg_proc = None
        
        self.speech_buffer = []
        self.recording = False
        self.session_start = None
        self.total_samples_processed = 0
        self.segment_start_samples = 0
        self.last_segment_end_samples = 0 # Track to prevent lookback overlap
        self.last_speech_sample = 0       # Track exact end of speech for accurate anchoring
        self.draft_history = []  # List of (draft_text, draft_end_samples, buffer_size) tuples
        
        # Buffer to hold raw PCM bytes for the current archive segment
        self.pcm_archive_buffer = bytearray()
        self.samples_at_last_archive = 0
        
        self.last_refresh_time = time.time()
        self.lookback_buffer = deque(maxlen=config.LOOKBACK_CHUNKS)

    def save_archive_chunk(self):
        """Encodes the current PCM buffer to MP3 and saves it, anchored to the stream start."""
        if not self.pcm_archive_buffer or self.session_start is None:
            return
            
        session_str = self.session_start.strftime("%Y%m%d_%H%M%S_%f")
        # Use exact sample offset for deterministic naming
        filename = f"kqed_{session_str}_{self.samples_at_last_archive:012d}.mp3"
        filepath = config.AUDIO_DIR / filename
        
        try:
            # Use ffmpeg to encode the accumulated PCM bytes to MP3
            cmd = [
                "ffmpeg",
                "-y",                    # Overwrite if exists
                "-f", "s16le",
                "-ar", str(config.VAD_SAMPLING_RATE),
                "-ac", "1",
                "-i", "pipe:0",          # Input from stdin
                "-f", "mp3",
                "-b:a", "64k",
                str(filepath)
            ]
            
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            proc.communicate(input=self.pcm_archive_buffer)
            
            logger.info(f"Archived audio: {filename} ({len(self.pcm_archive_buffer)/(1024*1024):.2f} MB PCM)")
            
            self.pcm_archive_buffer = bytearray() 
            self.samples_at_last_archive = self.total_samples_processed
        except Exception as e:
            logger.error(f"Failed to save archive: {e}")


    def align_words_to_timestamps(self, final_text, segment_end_samples):
        """
        Align words in final text using backwards-looking anchors.
        For each word, find the last draft that doesn't contain it - that sample count
        marks the beginning of the word.
        
        Args:
            final_text: Final transcription text
            segment_end_samples: Current sample count (end of segment)
        
        Returns:
            Dict mapping word_index -> (start_samples, end_samples) tuples
        """
        final_words = final_text.split()
        if not final_words:
            return {}
        
        # Normalize all words for matching
        final_normalized = [normalize_word(w) for w in final_words]
        
        # Track start timestamp for each word
        # word_index -> start_sample_count
        word_timestamps = {}
        
        # Filter drafts to only include those valid for current segment
        # (draft_end_samples must be >= segment_start_samples)
        valid_drafts = [
            draft for draft in self.draft_history
            if (len(draft) == 2 and draft[1] >= self.segment_start_samples) or
               (len(draft) == 3 and draft[1] >= self.segment_start_samples)
        ]
        
        if not valid_drafts:
            return {}
        
        # Normalize and prepare all drafts for backwards search
        # Sort by sample count (newest first for backwards search)
        sorted_drafts = sorted(valid_drafts, key=lambda x: x[1], reverse=True)
        
        prepared_drafts = []
        for draft_entry in sorted_drafts:
            # Handle both old format (2-tuple) and new format (3-tuple)
            if len(draft_entry) == 2:
                draft_text, draft_end_samples = draft_entry
                buffer_size = draft_end_samples - self.segment_start_samples
                if buffer_size < 0:
                    continue
            else:
                draft_text, draft_end_samples, buffer_size = draft_entry
                if draft_end_samples < self.segment_start_samples:
                    continue
                buffer_size = draft_end_samples - self.segment_start_samples
            
            draft_words = draft_text.split()
            draft_normalized = [normalize_word(w) for w in draft_words]
            prepared_drafts.append({
                'text': draft_text,
                'words': draft_words,
                'normalized': draft_normalized,
                'end_samples': draft_end_samples,
                'buffer_size': buffer_size
            })
        
        # For each word in final text, find when it first appears using backwards search
        # Process words in order to handle repeated words correctly using sequence alignment
        # CRITICAL: Ensure anchors are monotonic - each word's anchor must be >= previous word's anchor
        prev_anchor = self.segment_start_samples
        
        for word_idx in range(len(final_words)):
            if word_idx in word_timestamps:
                continue  # Already timestamped
            
            target_word_norm = final_normalized[word_idx]
            
            # For repeated words, we need to match in sequence context
            # Build a prefix of final text up to (but not including) word_idx
            # to match against drafts for proper sequence alignment
            prefix_normalized = final_normalized[:word_idx]
            
            # Go backwards through drafts (newest to oldest)
            # Find the last draft that doesn't contain this word at the correct position
            # Strategy: iterate backwards, and when we find a draft WITH the word,
            # continue to find the last (oldest) draft WITHOUT it
            # BUT: anchor must be >= prev_anchor to maintain monotonicity
            last_draft_without_word = None
            found_draft_with_word = False
            
            for draft in prepared_drafts:
                draft_normalized = draft['normalized']
                
                # Check if this draft contains the word at the correct position in sequence
                # For repeated words, we need to match the sequence up to this point
                contains_word_at_position = False
                
                if len(draft_normalized) > 0:
                    # Try to align the prefix sequence to find where word_idx would appear
                    # Use a more robust sequence alignment that handles:
                    # - Extra words in draft
                    # - Missing words in draft
                    # - Repeated words
                    
                    # For each possible position in the draft where our target word appears
                    for draft_pos in range(len(draft_normalized)):
                        if draft_normalized[draft_pos] == target_word_norm:
                            # Check if the prefix before this position aligns with our prefix
                            # Use a flexible matching that allows for extra/missing words
                            
                            # Match prefix using a greedy alignment from the end
                            final_prefix_idx = len(prefix_normalized) - 1
                            draft_check_idx = draft_pos - 1
                            matched_count = 0
                            
                            # Match backwards, allowing draft to have extra words
                            while final_prefix_idx >= 0 and draft_check_idx >= 0:
                                if prefix_normalized[final_prefix_idx] == draft_normalized[draft_check_idx]:
                                    matched_count += 1
                                    final_prefix_idx -= 1
                                    draft_check_idx -= 1
                                else:
                                    # Draft has extra word, skip it and continue matching
                                    draft_check_idx -= 1
                            
                            # If we matched all words in prefix, this is a valid position
                            # This means the draft contains the sequence up to our target word
                            if matched_count == len(prefix_normalized) and len(prefix_normalized) > 0:
                                contains_word_at_position = True
                                break
                            # Special case: if prefix is empty (first word), any occurrence is valid
                            elif len(prefix_normalized) == 0:
                                contains_word_at_position = True
                                break
                    
                    # Additional check: if prefix is empty and word appears anywhere, it's valid
                    if len(prefix_normalized) == 0 and target_word_norm in draft_normalized:
                        contains_word_at_position = True
                
                if not contains_word_at_position:
                    # This draft doesn't contain the word at the correct position
                    # If we've already found a draft with the word, this is the last one without it
                    if found_draft_with_word:
                        last_draft_without_word = draft
                        break  # Found our anchor point
                    # Otherwise, keep track of it in case the word never appears
                    last_draft_without_word = draft
                else:
                    # Found a draft with the word - mark that we've seen it
                    # Continue searching backwards to find the last draft without it
                    found_draft_with_word = True
            
            # Determine start timestamp
            if last_draft_without_word is not None:
                # Found a draft without the word - use its end_samples as the start
                # This is the "backwards anchor": the sample count where the word first appears
                anchor = last_draft_without_word['end_samples']
                # CRITICAL: Anchor must be STRICTLY > previous anchor to maintain monotonicity
                # Words appear sequentially, so later words must start AFTER earlier words
                # If anchor would be <= prev_anchor, increment to ensure strict monotonicity
                # This prevents multiple words from having the same start timestamp
                if anchor <= prev_anchor:
                    # Increment by 1 sample to ensure strict monotonicity
                    # This handles cases where backwards anchors find the same draft for multiple words
                    anchor = prev_anchor + 1
                word_timestamps[word_idx] = anchor
                prev_anchor = anchor
            else:
                # Word appears in all drafts (or no drafts) - use previous anchor or segment start
                # This handles edge cases where word was present from the beginning
                anchor = max(prev_anchor + 1, self.segment_start_samples)
                word_timestamps[word_idx] = anchor
                prev_anchor = anchor
        
        # Monotonicity is now enforced during anchor assignment above
        # No need for additional enforcement here
        
        # Build result with timestamps - return word_index -> (start_samples, end_samples) mapping
        # CRITICAL: Ensure end_samples is always > start_samples and timestamps are strictly increasing
        result = {}
        
        # First pass: assign start timestamps (already done in word_timestamps)
        # Second pass: assign end timestamps ensuring strict monotonicity
        for i, word in enumerate(final_words):
            if i in word_timestamps:
                start_samples = word_timestamps[i]
                
                # Find the next word with a DIFFERENT start timestamp
                # This ensures words with the same start get different ends
                end_samples = segment_end_samples
                for j in range(i + 1, len(final_words)):
                    if j in word_timestamps:
                        next_start = word_timestamps[j]
                        # Use next_start only if it's strictly greater than current start
                        if next_start > start_samples:
                            end_samples = next_start
                            break
                
                # CRITICAL: If no next word with different start, we must ensure end > start
                # With strict monotonicity enforcement, end_samples should always be > start_samples
                # But add safety check just in case
                if end_samples <= start_samples:
                    # This should never happen with proper strict monotonicity, but safety check
                    # Ensure end is at least 1 sample after start
                    if segment_end_samples > start_samples:
                        end_samples = segment_end_samples
                    else:
                        # Segment end is invalid - this shouldn't happen, skip this word
                        logger.error(f"Invalid segment: start={start_samples}, segment_end={segment_end_samples}")
                        continue
                
                result[i] = (start_samples, end_samples)
        
        return result

    def commit_segment(self, final_text, segment_end_samples):
        """
        Finalizes a segment by aligning the final high-accuracy text with the 
        draft observations collected during live transcription.
        Words without timestamps are appended to the previous word that has a timestamp.
        """
        final_text = final_text.strip()
        if not final_text:
            self.draft_history = []
            return
        
        final_words = final_text.split()
        if not final_words:
            self.draft_history = []
            return
        
        # Align words to timestamps using draft history
        # Returns dict: word_index -> (start_samples, end_samples)
        timestamp_map = self.align_words_to_timestamps(final_text, segment_end_samples)
        
        if not timestamp_map:
            # No words had timestamps - skip this segment
            self.draft_history = []
            return
        
        # Group words: words with the same start timestamp should be on the same line
        # Words without timestamps are appended to the previous group
        # End is derived from the next group's start (never same as start)
        
        i = 0
        
        while i < len(final_words):
            # Find the start of the current group
            # Skip words without timestamps - they'll be grouped with the next word that has one
            group_start_idx = i
            while group_start_idx < len(final_words) and group_start_idx not in timestamp_map:
                group_start_idx += 1
            
            if group_start_idx >= len(final_words):
                break  # No more words with timestamps
            
            # Current group starts at group_start_idx
            current_start, _ = timestamp_map[group_start_idx]
            
            # Collect all words from i to group_start_idx (words without timestamps at start)
            # and then all words with the same start timestamp
            group_words = []
            
            # First, add any words without timestamps before the first timestamped word
            for j in range(i, group_start_idx):
                group_words.append(final_words[j])
            
            # Now add words with the same start timestamp
            j = group_start_idx
            while j < len(final_words):
                if j in timestamp_map:
                    word_start, _ = timestamp_map[j]
                    if word_start != current_start:
                        # Different start timestamp - start of next group
                        break
                    # Same start timestamp - add to current group
                    group_words.append(final_words[j])
                    j += 1
                else:
                    # Word without timestamp - append to current group
                    group_words.append(final_words[j])
                    j += 1
            
            # Determine end_samples: use the start of the next group, or segment_end_samples
            # CRITICAL: end must be > start, and must be monotonic
            end_s = segment_end_samples
            if j < len(final_words):
                # Find the next word with a timestamp (start of next group)
                next_group_start_idx = j
                while next_group_start_idx < len(final_words) and next_group_start_idx not in timestamp_map:
                    next_group_start_idx += 1
                
                if next_group_start_idx < len(final_words):
                    next_start, _ = timestamp_map[next_group_start_idx]
                    # Only use next_start if it's greater than current_start
                    # This prevents backwards timestamps
                    if next_start > current_start:
                        end_s = next_start
            
            # CRITICAL: Ensure end is always greater than start
            # This is a final safety check to prevent invalid timestamps
            if end_s <= current_start:
                # If end would be <= start, use segment_end_samples
                # But also ensure segment_end_samples is actually greater
                if segment_end_samples <= current_start:
                    # This shouldn't happen, but if it does, skip this group
                    logger.warning(f"Skipping group with invalid timestamps: start={current_start}, segment_end={segment_end_samples}")
                    i = j
                    continue
                end_s = segment_end_samples
            
            # Final verification: end must be strictly greater than start
            if end_s <= current_start:
                logger.error(f"Invalid timestamp after all checks: start={current_start}, end={end_s}, segment_end={segment_end_samples}")
                i = j
                continue
            
            # Output the group
            text = " ".join(group_words)
            write_to_transcript(current_start, end_s, text)
            
            # Move to next group
            i = j
        
        self.last_segment_end_samples = segment_end_samples
        self.draft_history = []

    def signal_handler(self, signum, frame):
        logger.info(f"Signal {signum} received. Initiating graceful shutdown...")
        self.shutdown_event.set()

    def run(self):
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.ffmpeg_proc = run_ffmpeg(self.pcm_queue, self.shutdown_event)
        
        logger.info(f"Connected to stream for live transcription: {config.STREAM_URL}")
        try:
            with requests.get(config.STREAM_URL, stream=True, timeout=20) as r:
                r.raise_for_status()

                for chunk in r.iter_content(chunk_size=4096):
                    if self.shutdown_event.is_set(): break
                    if not chunk: continue
                    
                    # Feed raw stream to decoder
                    try:
                        self.ffmpeg_proc.stdin.write(chunk)
                        self.ffmpeg_proc.stdin.flush()
                    except BrokenPipeError: break

                    while not self.pcm_queue.empty():
                        pcm_bytes = self.pcm_queue.get()
                        # Convert bytes back to float32 for VAD/Transcriber
                        pcm_chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        num_samples = len(pcm_chunk)
                        
                        if self.session_start is None:
                            # Use full precision for session start to eliminate initial drift
                            self.session_start = datetime.now()
                            write_session_marker(self.session_start)
                        
                        # Collect raw PCM bytes for the archive - this is perfectly synced with num_samples
                        self.pcm_archive_buffer.extend(pcm_bytes)

                        # Strictly track sample boundaries for this chunk
                        chunk_start = self.total_samples_processed
                        chunk_end = chunk_start + num_samples

                        # Convert to torch tensor for Silero VAD which expects it
                        pcm_tensor = torch.from_numpy(pcm_chunk)
                        speech_prob = self.vad_model(pcm_tensor, config.VAD_SAMPLING_RATE).item()
                        
                        # Update ground truth for last active speech
                        if speech_prob > config.VAD_THRESHOLD:
                            self.last_speech_sample = chunk_end
                            
                        speech_dict = self.vad_iterator(pcm_tensor)
                        
                        if speech_dict:
                            # Start of speech detection
                            if "start" in speech_dict and not self.recording:
                                self.recording = True
                                # Use exact sample offset from VAD for start
                                start_sample = speech_dict["start"]
                                lookback_samples = sum(len(c) for c in self.lookback_buffer)
                                
                                # CLAMP LOOKBACK: Ensure we don't overlap with the previous segment
                                requested_start = start_sample - lookback_samples
                                self.segment_start_samples = max(requested_start, self.last_segment_end_samples)
                                
                                self.speech_buffer = list(self.lookback_buffer)
                                self.draft_history = []  # Reset draft history for new segment
                                self.last_refresh_time = time.time()
                            
                            # End of speech detection handled within self.recording block

                        if self.recording:
                            self.speech_buffer.append(pcm_chunk)
                            
                            # Determine if we should finalize this segment
                            is_vad_end = speech_dict and "end" in speech_dict
                            is_max_duration = (sum(len(c) for c in self.speech_buffer) / config.VAD_SAMPLING_RATE) > config.LIVE_MAX_SPEECH_S
                            
                            if is_vad_end or is_max_duration:
                                full_speech = np.concatenate(self.speech_buffer)
                                text = self.transcribe(full_speech)
                                
                                # Use the exact sample offset from VAD for end if available
                                final_boundary = speech_dict["end"] if is_vad_end else chunk_end
                                self.commit_segment(text, final_boundary)
                                
                                self.speech_buffer = []
                                self.vad_iterator.reset_states()
                                
                                if is_vad_end:
                                    self.recording = False
                            else:
                                # Continuous recording (Live feedback and draft collection)
                                if (time.time() - self.last_refresh_time) > config.LIVE_MIN_REFRESH_S:
                                    if speech_prob > config.VAD_THRESHOLD:
                                        full_speech = np.concatenate(self.speech_buffer)
                                        if len(full_speech) >= 2000:  # Minimum for Moonshine
                                            text = self.transcribe(full_speech)
                                            if text.strip():
                                                # Store draft with sample count at end of current speech buffer
                                                # and buffer size for accurate word position calculation
                                                buffer_samples = sum(len(c) for c in self.speech_buffer)
                                                draft_end_samples = self.segment_start_samples + buffer_samples
                                                buffer_size = buffer_samples
                                                self.draft_history.append((text, draft_end_samples, buffer_size))
                                                
                                                # Keep drafts from the current segment only
                                                # Filter out drafts that are before segment start
                                                self.draft_history = [
                                                    (dt, ds, bs) for dt, ds, bs in self.draft_history
                                                    if ds >= self.segment_start_samples
                                                ]
                                        
                                        self.last_refresh_time = time.time()

                        if not self.recording:
                            # Update lookback buffer only when not in an active speech segment
                            self.lookback_buffer.append(pcm_chunk)

                        # Advance the global sample counter
                        self.total_samples_processed = chunk_end
                        
                        if (self.total_samples_processed - self.samples_at_last_archive) >= config.SAMPLES_PER_ARCHIVE:
                            self.save_archive_chunk()

        except Exception as e:
            if not self.shutdown_event.is_set():
                logger.error(f"Error in live transcriber: {e}")
        finally:
            logger.info("Flushing remaining audio archive buffer...")
            self.save_archive_chunk()
            self.shutdown_event.set()
            if self.ffmpeg_proc: self.ffmpeg_proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moonshine/tiny", choices=["moonshine/base", "moonshine/tiny"])
    args = parser.parse_args()

    engine = LiveTranscriberEngine(args.model)
    engine.run()
