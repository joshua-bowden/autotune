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
from datetime import datetime
from queue import Queue
from collections import deque
from silero_vad import VADIterator, load_silero_vad

import config
from utils import setup_logging, Transcriber

logger = setup_logging(__name__)

def write_to_transcript(session_ts, start_samples, end_samples, text):
    """Append transcript line: YYMMDD_HHMMSS|start_samples|end_samples|text."""
    os.makedirs(config.TRANSCRIPT_DIR, exist_ok=True)
    with open(config.TRANSCRIPT_DIR / "current_transcript.txt", "a", encoding="utf-8") as f:
        f.write(f"{session_ts}|{start_samples}|{end_samples}|{text.strip()}\n")

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
        # Silero VADIterator reports sample indices relative to its last reset.
        # Track the absolute stream sample offset corresponding to iterator index 0.
        self.vad_base_samples = 0
        
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


    def commit_segment(self, start_samples, end_samples, final_text):
        """
        Finalizes a segment by writing the full inference with timing info.
        """
        final_text = final_text.strip()
        if not final_text:
            return
        if end_samples <= start_samples:
            return

        session_ts = self.session_start.strftime("%y%m%d_%H%M%S") if self.session_start else datetime.now().strftime("%y%m%d_%H%M%S")
        write_to_transcript(session_ts, start_samples, end_samples, final_text)
        self.last_segment_end_samples = end_samples

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
                                vad_start_abs = self.vad_base_samples + int(speech_dict["start"])
                                lookback_samples = sum(len(c) for c in self.lookback_buffer)
                                
                                # CLAMP LOOKBACK: Ensure we don't overlap with the previous segment
                                requested_start = vad_start_abs - lookback_samples
                                self.segment_start_samples = max(requested_start, self.last_segment_end_samples)
                                
                                self.speech_buffer = list(self.lookback_buffer)
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
                                if is_vad_end and speech_dict and "end" in speech_dict:
                                    final_boundary = self.vad_base_samples + int(speech_dict["end"])
                                else:
                                    final_boundary = chunk_end

                                # Safety: never allow end <= start in the transcript/metadata
                                if final_boundary <= self.segment_start_samples:
                                    final_boundary = chunk_end
                                self.commit_segment(self.segment_start_samples, final_boundary, text)
                                
                                # If we cut only because of max-duration (no VAD end),
                                # keep recording but advance the base start so subsequent
                                # segments don't reuse the same start sample.
                                if is_max_duration and not is_vad_end:
                                    self.segment_start_samples = final_boundary
                                
                                self.speech_buffer = []
                                self.vad_iterator.reset_states()
                                # Iterator indices now restart at 0 for the next chunk.
                                self.vad_base_samples = chunk_end
                                
                                if is_vad_end:
                                    self.recording = False
                            else:
                                # Continuous recording (Live feedback and draft collection)
                                if (time.time() - self.last_refresh_time) > config.LIVE_MIN_REFRESH_S:
                                    # Only emit final segments (no drafts)
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
