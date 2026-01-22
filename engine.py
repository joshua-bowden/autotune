"""
Unified Streaming Engine for the Radio Archiver.

This module handles:
1. Long-lived connection to the radio stream.
2. Real-time decoding of MP3 to PCM via FFmpeg.
3. Voice Activity Detection (VAD) via Silero.
4. Transcription via Moonshine.
5. Rolling 1-minute MP3 archives.
"""

import logging
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty
from collections import deque
from typing import Optional, List

import numpy as np
import requests
import torch
from silero_vad import VADIterator, load_silero_vad
from transformers import AutoProcessor, MoonshineForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

import config
from utils import setup_logging

logger = setup_logging(__name__)

# Local constants derived from config
SAMPLING_RATE = config.VAD_SAMPLING_RATE
CHUNK_SIZE = config.VAD_CHUNK_SIZE
ARCHIVE_INTERVAL = config.CHUNK_DURATION
LOOKBACK_CHUNKS = config.LOOKBACK_CHUNKS

class StreamingEngine:
    def __init__(self):
        self.shutdown_requested = False
        self.pcm_reader_alive = True
        self.stream_url = config.STREAM_URL
        
        # Models
        logger.info("Loading models...")
        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(
            model=self.vad_model,
            sampling_rate=SAMPLING_RATE,
            threshold=config.VAD_THRESHOLD,
            min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
        )
        
        # Moonshine
        model_id = "UsefulSensors/moonshine-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MoonshineForConditionalGeneration.from_pretrained(model_id)
        logger.info("Models loaded.")

        # Buffers and Queues
        self.mp3_buffer = bytearray()
        self.pcm_queue = Queue()
        self.transcription_queue = Queue() # New queue for background transcription
        
        # Timing
        self.session_start = None # Will be set on first chunk arrival
        self.samples_processed = 0
        self.samples_at_last_archive = 0
        
        # Transcript file
        self.transcript_file = config.TRANSCRIPT_DIR / "current_transcript.txt"
        
        # Parallel Transcription State
        self.sequence_id = 0
        self.next_to_commit = 0
        self.results_cache = {} # id -> (sentences, start, end)
        self.cache_lock = threading.Lock()
        
        # Optimization: Moonshine on CPU
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2)) 
        
        # Start transcription manager
        self._start_transcription_manager()

    def signal_handler(self, signum, frame):
        logger.info("Shutdown requested...")
        self.shutdown_requested = True

    def run_ffmpeg(self):
        """Pipes raw stream bytes to ffmpeg and reads back PCM."""
        cmd = [
            "ffmpeg",
            "-i", "pipe:0",          # Input from stdin
            "-f", "s16le",          # Output format raw pcm 16-bit
            "-ar", str(SAMPLING_RATE),
            "-ac", "1",             # Mono
            "pipe:1"                # Output to stdout
        ]
        
        self.ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture stderr for debugging
            bufsize=1024 * 1024
        )
        
        # Thread to log ffmpeg errors
        def log_ffmpeg_stderr():
            for line in iter(self.ffmpeg_proc.stderr.readline, b''):
                logger.debug(f"FFmpeg: {line.decode().strip()}")
        threading.Thread(target=log_ffmpeg_stderr, daemon=True).start()
        
        # Thread to read PCM from ffmpeg stdout
        def read_pcm():
            # Local buffer to ensure we only pass chunks of exactly CHUNK_SIZE
            pcm_buffer = bytearray()
            byte_chunk_size = CHUNK_SIZE * 2 # 16-bit = 2 bytes per sample
            
            while not self.shutdown_requested:
                try:
                    # Read from pipe - might return less than requested
                    chunk = self.ffmpeg_proc.stdout.read(byte_chunk_size)
                    if not chunk:
                        break
                    
                    pcm_buffer.extend(chunk)
                    
                    # Process as many full chunks as we have
                    while len(pcm_buffer) >= byte_chunk_size:
                        to_process = pcm_buffer[:byte_chunk_size]
                        pcm_buffer = pcm_buffer[byte_chunk_size:]
                        
                        # Convert to float32 expected by VAD/Moonshine
                        samples = np.frombuffer(to_process, dtype=np.int16).astype(np.float32) / 32768.0
                        self.pcm_queue.put(samples)
                except Exception as e:
                    logger.error(f"Error reading PCM: {e}")
                    break
            self.pcm_reader_alive = False
            logger.info("PCM reader thread stopped.")

        threading.Thread(target=read_pcm, daemon=True).start()

    def _start_transcription_manager(self):
        """Starts a pool of transcription workers and one ordered committer."""
        num_workers = config.TRANSCRIPTION_WORKERS 
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Start the ordered committer thread
        threading.Thread(target=self._ordered_committer, daemon=True).start()

    def _transcription_task(self, seq_id: int, speech: np.ndarray, start_sec: float, end_sec: float):
        """Worker task that transcribes and caches the result."""
        try:
            duration = len(speech) / SAMPLING_RATE
            # logger.info(f"Worker: Starting transcription for seg {seq_id} ({duration:.1f}s)")
            text = self.transcribe(speech)
            sentences = []
            if text and text.strip():
                sentences = self.split_into_sentences(text)
            
            with self.cache_lock:
                self.results_cache[seq_id] = (sentences, start_sec, end_sec)
            # logger.info(f"Worker: Finished seg {seq_id} ({len(sentences)} sentences)")
        except Exception as e:
            logger.error(f"Error in transcription task {seq_id}: {e}")
            with self.cache_lock:
                self.results_cache[seq_id] = ([], start_sec, end_sec)

    def _ordered_committer(self):
        """Thread that writes results to the file in strict sequence order."""
        logger.info("Ordered committer started.")
        last_committer_heartbeat = time.time()
        while not self.shutdown_requested:
            with self.cache_lock:
                if self.next_to_commit in self.results_cache:
                    sentences, start, end = self.results_cache.pop(self.next_to_commit)
                else:
                    sentences = None

            if sentences:
                self.save_sentences(sentences, start, end)
                self.next_to_commit += 1
                continue # Check for the next one immediately
                
                # Heartbeat: Log what we are waiting for periodically
                if time.time() - last_committer_heartbeat > config.ENGINE_HEARTBEAT_INTERVAL_S:
                    logger.info(f"Committer: Still waiting for segment {self.next_to_commit} (Cache size: {len(self.results_cache)})")
                    last_committer_heartbeat = time.time()
            time.sleep(1.0) # Check periodically

    def soft_reset(self):
        """Soft resets the Silero VADIterator state."""
        self.vad_iterator.triggered = False
        self.vad_iterator.temp_end = 0
        self.vad_iterator.current_sample = 0

    def save_sentences(self, sentences: List[str], start_sec: float, end_sec: float):
        """Saves sentences with distributed timestamps."""
        def get_precision_time(seconds_offset):
            dt = self.session_start + timedelta(seconds=seconds_offset)
            date_str = dt.strftime("%Y%m%d")
            clock_str = dt.strftime("%H%M%S") + f".{int((seconds_offset % 1) * 100):02d}"
            return date_str, clock_str
        
        total_chars = sum(len(s) for s in sentences)
        duration = end_sec - start_sec
        
        current_start = start_sec
        try:
            with open(self.transcript_file, "a", encoding="utf-8") as f:
                for sentence in sentences:
                    sentence_weight = len(sentence) / total_chars if total_chars > 0 else (1.0 / len(sentences))
                    sentence_duration = duration * sentence_weight
                    sentence_end = current_start + sentence_duration
                    
                    date_str, start_clock = get_precision_time(current_start)
                    _, end_clock = get_precision_time(sentence_end)
                    
                    f.write(f"{date_str}|{start_clock}|{end_clock}|{sentence}\n")
                    logger.info(f"[{start_clock}] {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
                    
                    current_start = sentence_end
        except IOError as e:
            logger.error(f"Failed to write transcript: {e}")

    def transcribe(self, speech: np.ndarray) -> Optional[str]:
        """Transcribe a speech segment using Moonshine."""
        try:
            inputs = self.processor(
                speech,
                sampling_rate=SAMPLING_RATE,
                return_tensors="pt"
            )
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex-based logic."""
        # Split on . ! ? followed by space or end of string
        # Keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def save_archive_chunk(self):
        """Saves the current MP3 buffer with a filename anchored to the stream start."""
        if not self.mp3_buffer or self.session_start is None:
            return
            
        # The start time of this chunk is session_start + samples_at_last_archive
        chunk_start_dt = self.session_start + timedelta(seconds=self.samples_at_last_archive / SAMPLING_RATE)
        timestamp_str = chunk_start_dt.strftime("%Y%m%d_%H%M%S")
        
        filename = f"kqed_{timestamp_str}.mp3"
        filepath = config.AUDIO_DIR / filename
        
        try:
            with open(filepath, "wb") as f:
                f.write(self.mp3_buffer)
            logger.info(f"Archived audio: {filename} ({len(self.mp3_buffer)/(1024*1024):.2f} MB)")
            
            # No overlap. Bit-perfect contiguity ensures zero timing drift.
            self.mp3_buffer = bytearray() 
            self.samples_at_last_archive = self.samples_processed
        except IOError as e:
            logger.error(f"Failed to save archive: {e}")

    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.run_ffmpeg()
        
        logger.info(f"Connecting to stream: {self.stream_url}")
        try:
            with requests.get(self.stream_url, stream=True, timeout=20) as r:
                r.raise_for_status()
                
                speech_buffer = []
                recording = False
                speech_start_sec = 0.0
                lookback_buffer = deque(maxlen=LOOKBACK_CHUNKS)
                
                # No local counter needed, using self.samples_processed

                last_heartbeat = time.time()

                for chunk in r.iter_content(chunk_size=4096):
                    if self.shutdown_requested:
                        break
                    
                    if not chunk:
                        continue
                        
                    if time.time() - last_heartbeat > config.ENGINE_HEARTBEAT_INTERVAL_S:
                        pending = self.sequence_id - self.next_to_commit
                        logger.info(f"Engine heartbeat: {self.samples_processed/SAMPLING_RATE:.1f}s processed, {pending} segments pending, PCM reader: {'OK' if self.pcm_reader_alive else 'FAILED'}")
                        last_heartbeat = time.time()
                        
                    # Initialize start times on first chunk
                    if self.session_start is None:
                        # Use floor to nearest second for cleaner filenames/math
                        now = datetime.now()
                        self.session_start = now.replace(microsecond=0)
                        logger.info(f"Stream session started at {self.session_start}")

                    # 1. Archive raw bytes
                    self.mp3_buffer.extend(chunk)
                    
                    # 2. Feed decoder
                    try:
                        self.ffmpeg_proc.stdin.write(chunk)
                        self.ffmpeg_proc.stdin.flush()
                    except BrokenPipeError:
                        logger.error("FFmpeg pipe broken.")
                        break
                    
                    # 3. Process any available PCM
                    while not self.pcm_queue.empty():
                        pcm_chunk = self.pcm_queue.get()
                        
                        if not recording:
                            lookback_buffer.append(pcm_chunk)

                        # VAD logic
                        speech_dict = self.vad_iterator(pcm_chunk)
                        
                        if speech_dict:
                            if "start" in speech_dict:
                                recording = True
                                # Adjust start time backwards by the size of the lookback buffer
                                lookback_samples = sum(len(c) for c in lookback_buffer)
                                speech_start_sec = (self.samples_processed - lookback_samples) / SAMPLING_RATE
                                logger.info(f"VAD: Speech start detected (including {lookback_samples} samples lookback) at {speech_start_sec:.2f}s")
                                # Seed buffer with lookback for better leading consonant capture
                                speech_buffer = list(lookback_buffer)
                            
                            if "end" in speech_dict and recording:
                                recording = False
                                speech_end_sec = self.samples_processed / SAMPLING_RATE
                                if speech_buffer:
                                    full_speech = np.concatenate(speech_buffer)
                                    logger.info(f"VAD: Speech end. Submitting seg {self.sequence_id} ({speech_end_sec - speech_start_sec:.1f}s)")
                                    self.executor.submit(
                                        self._transcription_task,
                                        self.sequence_id,
                                        full_speech,
                                        speech_start_sec,
                                        speech_end_sec
                                    )
                                    self.sequence_id += 1
                                
                                speech_buffer = []
                                self.soft_reset()
                        
                        if recording:
                            speech_buffer.append(pcm_chunk)
                            # Safety: Max segment length
                            current_duration = (self.samples_processed / SAMPLING_RATE) - speech_start_sec
                            if current_duration > config.MAX_SEGMENT_DURATION_S:
                                logger.info(f"VAD: Max segment length reached ({config.MAX_SEGMENT_DURATION_S}s). Splitting.")
                                full_speech = np.concatenate(speech_buffer)
                                self.executor.submit(
                                    self._transcription_task,
                                    self.sequence_id,
                                    full_speech,
                                    speech_start_sec,
                                    self.samples_processed / SAMPLING_RATE
                                )
                                self.sequence_id += 1
                                speech_buffer = []
                                speech_start_sec = self.samples_processed / SAMPLING_RATE # Reset start for next part
                                self.soft_reset()
                        
                        self.samples_processed += len(pcm_chunk)
                        
                        # Trigger archive every ARCHIVE_INTERVAL seconds, rounded UP to nearest MP3 frame.
                        samples_per_archive = config.SAMPLES_PER_ARCHIVE 
                        if (self.samples_processed - self.samples_at_last_archive) >= samples_per_archive:
                            self.save_archive_chunk()

        except Exception as e:
            logger.error(f"Engine loop error: {e}")
        finally:
            self.save_archive_chunk()
            if self.ffmpeg_proc:
                self.ffmpeg_proc.terminate()
            logger.info("Engine stopped.")

if __name__ == "__main__":
    engine = StreamingEngine()
    engine.run()
