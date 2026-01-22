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
import numpy as np
from datetime import datetime, timedelta
from queue import Queue
from silero_vad import VADIterator, load_silero_vad
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
import live_constants as lc

class Transcriber:
    def __init__(self, model_name):
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.tokenizer = load_tokenizer()
        self.rate = lc.SAMPLING_RATE
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
    
    # os.makedirs(lc.TRANSCRIPT_DIR, exist_ok=True)
    # with open(lc.TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
    #     f.write(f"{date_str}|{start_clock}|{end_clock}|{text.strip()}\n")
    pass

def run_ffmpeg(pcm_queue, shutdown_event):
    """Pipes raw stream bytes to ffmpeg and reads back PCM."""
    cmd = [
        "ffmpeg",
        "-i", "pipe:0",          # Input from stdin
        "-f", "s16le",          # Output format raw pcm 16-bit
        "-ar", str(lc.SAMPLING_RATE),
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
        byte_chunk_size = lc.CHUNK_SIZE * 2
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

caption_cache = []

def print_captions(text):
    """Prints right justified on same line, prepending cached captions."""
    width = lc.MAX_LINE_LENGTH
    if len(text) < width:
        for caption in caption_cache[::-1]:
            text = caption + " " + text
            if len(text) > width:
                break
    if len(text) > width:
        text = text[-width:]
    else:
        text = " " * (width - len(text)) + text
    print("\r" + (" " * width) + "\r" + text, end="", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=lc.MOONSHINE_MODEL, choices=["moonshine/base", "moonshine/tiny"])
    args = parser.parse_args()

    transcribe = Transcriber(args.model)
    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=lc.SAMPLING_RATE,
        threshold=0.5,
        min_silence_duration_ms=400,
    )

    pcm_queue = Queue()
    shutdown_event = threading.Event()
    ffmpeg_proc = run_ffmpeg(pcm_queue, shutdown_event)

    speech_buffer = np.empty(0, dtype=np.float32)
    recording = False
    
    session_start = None
    total_samples_processed = 0
    
    # State for current speech segment
    segment_start_samples = 0
    period_timestamps = []  
    last_period_count = 0
    
    last_refresh_time = time.time()

    def commit_segment(final_text, segment_end_samples):
        nonlocal segment_start_samples, period_timestamps
        if not final_text.strip():
            return
            
        # Add to display cache
        caption_cache.append(final_text.strip())
        
        sentences = [s.strip() + "." for s in final_text.split(".") if s.strip()]
        if not sentences:
            return

        current_start = segment_start_samples
        
        for i, sentence in enumerate(sentences):
            if i < len(period_timestamps):
                end_samples = period_timestamps[i]
            else:
                remaining = len(sentences) - i
                progress = (segment_end_samples - current_start) / remaining
                end_samples = int(current_start + progress)

            if end_samples <= current_start:
                end_samples = current_start + 1

            start_dt = session_start + timedelta(seconds=current_start / lc.SAMPLING_RATE)
            end_dt = session_start + timedelta(seconds=end_samples / lc.SAMPLING_RATE)
            
            write_to_transcript(start_dt, end_dt, sentence)
            current_start = end_samples
        
        segment_start_samples = segment_end_samples
        period_timestamps.clear()

    try:
        with requests.get(lc.STREAM_URL, stream=True, timeout=20) as r:
            r.raise_for_status()

            for chunk in r.iter_content(chunk_size=4096):
                if not chunk: continue
                
                try:
                    ffmpeg_proc.stdin.write(chunk)
                    ffmpeg_proc.stdin.flush()
                except BrokenPipeError:
                    break

                while not pcm_queue.empty():
                    pcm_chunk = pcm_queue.get()
                    if session_start is None:
                        session_start = datetime.now()
                    
                    total_samples_processed += len(pcm_chunk)
                    
                    speech_dict = vad_iterator(pcm_chunk)
                    
                    if speech_dict:
                        if "start" in speech_dict and not recording:
                            recording = True
                            segment_start_samples = total_samples_processed
                            speech_buffer = np.empty(0, dtype=np.float32)
                            period_timestamps = []
                            last_period_count = 0
                            last_refresh_time = time.time()
                        
                        if "end" in speech_dict and recording:
                            recording = False
                            text = transcribe(speech_buffer)
                            commit_segment(text, total_samples_processed)
                            print_captions("") # Finalize visual
                            vad_iterator.reset_states()

                    if recording:
                        speech_buffer = np.concatenate((speech_buffer, pcm_chunk))
                        
                        if (time.time() - last_refresh_time) > lc.MIN_REFRESH_SECS:
                            text = transcribe(speech_buffer)
                            print_captions(text)
                            
                            current_period_count = text.count(".")
                            if current_period_count > last_period_count:
                                for _ in range(current_period_count - last_period_count):
                                    period_timestamps.append(total_samples_processed)
                                last_period_count = current_period_count
                            
                            last_refresh_time = time.time()

                        if (len(speech_buffer) / lc.SAMPLING_RATE) > lc.MAX_SPEECH_SECS:
                            text = transcribe(speech_buffer)
                            commit_segment(text, total_samples_processed)
                            recording = False 
                            print_captions("")
                            # Soft reset logic
                            vad_iterator.triggered = False
                            vad_iterator.temp_end = 0
                            vad_iterator.current_sample = 0

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        shutdown_event.set()
        ffmpeg_proc.terminate()

if __name__ == "__main__":
    main()
