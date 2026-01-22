"""
Live transcription from radio stream with natural settlement and tracked period timestamps.
Uses Moonshine and SileroVAD ONNX models.
"""

import argparse
import os
import time
import subprocess
import threading
import requests
import logging
import numpy as np
from datetime import datetime, timedelta
from queue import Queue
from collections import deque
from silero_vad import VADIterator, load_silero_vad
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

import config
from utils import setup_logging

logger = setup_logging(__name__)

class Transcriber:
    def __init__(self, model_name):
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.tokenizer = load_tokenizer()
        self.rate = config.VAD_SAMPLING_RATE
        # Warmup
        self.model.generate(np.zeros((1, int(self.rate)), dtype=np.float32))

    def __call__(self, speech):
        """Returns transcription string."""
        # Moonshine expects a minimum amount of audio to process through its convolutions
        if len(speech) < 2000: # ~125ms minimum
            return ""
        try:
            tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
            text = self.tokenizer.decode_batch(tokens)[0]
            return text
        except Exception as e:
            # Fallback for any other ONNX shape errors
            return ""

def format_timestamp(dt):
    """Format datetime as YYYYMMDD|HHMMSS.ss"""
    date_str = dt.strftime("%Y%m%d")
    clock_str = dt.strftime("%H%M%S") + f".{dt.microsecond // 10000:02d}"
    return date_str, clock_str

def write_to_transcript(start_dt, end_dt, text):
    """Appends to the transcript file in the specific format."""
    date_str, start_clock = format_timestamp(start_dt)
    _, end_clock = format_timestamp(end_dt)
    
    os.makedirs(config.TRANSCRIPT_DIR, exist_ok=True)
    with open(config.TRANSCRIPT_DIR / "current_transcript.txt", "a", encoding="utf-8") as f:
        f.write(f"{date_str}|{start_clock}|{end_clock}|{text.strip()}\n")

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
                    samples = np.frombuffer(to_process, dtype=np.int16).astype(np.float32) / 32768.0
                    pcm_queue.put(samples)
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
        self.anchor_points = []
        
        self.mp3_buffer = bytearray()
        self.samples_at_last_archive = 0
        
        self.last_refresh_time = time.time()
        self.lookback_buffer = deque(maxlen=config.LOOKBACK_CHUNKS)

    def save_archive_chunk(self):
        """Saves the current MP3 buffer with a filename anchored to the stream start."""
        if not self.mp3_buffer or self.session_start is None:
            return
            
        chunk_start_dt = self.session_start + timedelta(seconds=self.samples_at_last_archive / config.VAD_SAMPLING_RATE)
        timestamp_str = chunk_start_dt.strftime("%Y%m%d_%H%M%S")
        
        filename = f"kqed_{timestamp_str}.mp3"
        filepath = config.AUDIO_DIR / filename
        
        try:
            with open(filepath, "wb") as f:
                f.write(self.mp3_buffer)
            logger.info(f"Archived audio: {filename} ({len(self.mp3_buffer)/(1024*1024):.2f} MB)")
            
            self.mp3_buffer = bytearray() 
            self.samples_at_last_archive = self.total_samples_processed
        except IOError as e:
            logger.error(f"Failed to save archive: {e}")

    def commit_segment(self, final_text, segment_end_samples):
        final_text = final_text.strip()
        if not final_text:
            self.anchor_points = []
            return
            
        final_words = final_text.split()
        valid_anchors = []
        last_pos, last_ts = -1, -1
        
        for pos, word, ts in self.anchor_points:
            if pos < len(final_words) and final_words[pos] == word:
                if pos > last_pos and ts > last_ts:
                    valid_anchors.append((pos, ts))
                    last_pos, last_ts = pos, ts

        curr_start_samples = self.segment_start_samples
        curr_word_idx = 0
        
        for pos, ts in valid_anchors:
            chunk_text = " ".join(final_words[curr_word_idx : pos + 1])
            if chunk_text.strip():
                start_dt = self.session_start + timedelta(seconds=curr_start_samples / config.VAD_SAMPLING_RATE)
                end_dt = self.session_start + timedelta(seconds=ts / config.VAD_SAMPLING_RATE)
                write_to_transcript(start_dt, end_dt, chunk_text)
            curr_start_samples, curr_word_idx = ts, pos + 1
            
        if curr_word_idx < len(final_words):
            chunk_text = " ".join(final_words[curr_word_idx:])
            if chunk_text.strip():
                start_dt = self.session_start + timedelta(seconds=curr_start_samples / config.VAD_SAMPLING_RATE)
                end_dt = self.session_start + timedelta(seconds=segment_end_samples / config.VAD_SAMPLING_RATE)
                write_to_transcript(start_dt, end_dt, chunk_text)

        self.segment_start_samples = segment_end_samples
        self.anchor_points = []

    def run(self):
        self.ffmpeg_proc = run_ffmpeg(self.pcm_queue, self.shutdown_event)
        
        logger.info(f"Connecting to stream for live transcription: {config.STREAM_URL}")
        try:
            with requests.get(config.STREAM_URL, stream=True, timeout=20) as r:
                r.raise_for_status()

                for chunk in r.iter_content(chunk_size=4096):
                    if self.shutdown_event.is_set(): break
                    if not chunk: continue
                    
                    self.mp3_buffer.extend(chunk)
                    try:
                        self.ffmpeg_proc.stdin.write(chunk)
                        self.ffmpeg_proc.stdin.flush()
                    except BrokenPipeError: break

                    while not self.pcm_queue.empty():
                        pcm_chunk = self.pcm_queue.get()
                        if self.session_start is None:
                            self.session_start = datetime.now()
                        
                        if not self.recording:
                            self.lookback_buffer.append(pcm_chunk)

                        speech_dict = self.vad_iterator(pcm_chunk)
                        
                        if speech_dict:
                            if "start" in speech_dict and not self.recording:
                                self.recording = True
                                lookback_samples = sum(len(c) for c in self.lookback_buffer)
                                self.segment_start_samples = self.total_samples_processed - lookback_samples
                                self.speech_buffer = list(self.lookback_buffer)
                                self.anchor_points = []
                                self.last_refresh_time = time.time()
                            
                            if "end" in speech_dict and self.recording:
                                self.recording = False
                                full_speech = np.concatenate(self.speech_buffer)
                                text = self.transcribe(full_speech)
                                self.commit_segment(text, self.total_samples_processed)
                                self.speech_buffer = []
                                self.vad_iterator.reset_states()

                        if self.recording:
                            self.speech_buffer.append(pcm_chunk)
                            if (time.time() - self.last_refresh_time) > config.LIVE_MIN_REFRESH_S:
                                current_ts = self.total_samples_processed
                                full_speech = np.concatenate(self.speech_buffer)
                                text = self.transcribe(full_speech)
                                words = text.split()
                                if words:
                                    self.anchor_points.append((len(words) - 1, words[-1], current_ts))
                                self.last_refresh_time = time.time()

                            if (sum(len(c) for c in self.speech_buffer) / config.VAD_SAMPLING_RATE) > config.LIVE_MAX_SPEECH_S:
                                full_speech = np.concatenate(self.speech_buffer)
                                text = self.transcribe(full_speech)
                                self.commit_segment(text, self.total_samples_processed)
                                # Start next segment immediately with empty buffer
                                self.speech_buffer = []
                                self.vad_iterator.reset_states()
                        
                        self.total_samples_processed += len(pcm_chunk)
                        
                        if (self.total_samples_processed - self.samples_at_last_archive) >= config.SAMPLES_PER_ARCHIVE:
                            self.save_archive_chunk()

        except KeyboardInterrupt:
            logger.info("\nStopping live transcriber...")
        finally:
            self.save_archive_chunk()
            self.shutdown_event.set()
            if self.ffmpeg_proc: self.ffmpeg_proc.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moonshine/tiny", choices=["moonshine/base", "moonshine/tiny"])
    args = parser.parse_args()

    engine = LiveTranscriberEngine(args.model)
    engine.run()
