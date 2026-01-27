"""Live captions from KQED stream using Moonshine and SileroVAD ONNX models."""

import argparse
import os
import signal
import time
import subprocess
import threading
import requests
from queue import Queue
from collections import deque
from pathlib import Path

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

import config
from utils import setup_logging
from pydub import AudioSegment

logger = setup_logging(__name__)

SAMPLING_RATE = config.VAD_SAMPLING_RATE
CHUNK_SIZE = config.VAD_CHUNK_SIZE
LOOKBACK_CHUNKS = config.LOOKBACK_CHUNKS

# These affect live caption updating - adjust for your platform speed and model.
MAX_SPEECH_SECS = config.LIVE_MAX_SPEECH_S
MIN_REFRESH_SECS = config.LIVE_MIN_REFRESH_S


class Transcriber(object):
    def __init__(self, model_name, rate=16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.rate = rate
        self.tokenizer = load_tokenizer()

        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        self.__call__(np.zeros(int(rate), dtype=np.float32))  # Warmup.

    def __call__(self, speech):
        """Returns string containing Moonshine transcription of speech."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]

        self.inference_secs += time.time() - start_time
        return text


def run_ffmpeg(pcm_queue, shutdown_event):
    """Pipes raw stream bytes to ffmpeg and reads back PCM."""
    cmd = [
        "ffmpeg",
        "-i", "pipe:0",          # Input from stdin
        "-f", "s16le",          # Output format raw pcm 16-bit
        "-ar", str(SAMPLING_RATE),
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
        byte_chunk_size = CHUNK_SIZE * 2
        while not shutdown_event.is_set():
            try:
                chunk = proc.stdout.read(byte_chunk_size)
                if not chunk:
                    break
                pcm_buffer.extend(chunk)
                while len(pcm_buffer) >= byte_chunk_size:
                    to_process = pcm_buffer[:byte_chunk_size]
                    pcm_buffer = pcm_buffer[byte_chunk_size:]
                    pcm_queue.put(to_process)
            except Exception:
                break
        proc.terminate()

    threading.Thread(target=read_pcm, daemon=True).start()
    return proc


def save_clipped_audio(audio_path: Path, clip_start_samples: int):
    """
    Save audio file clipped from a specific sample number to the end.
    
    Args:
        audio_path: Path to audio file in playground folder
        clip_start_samples: Sample number to start clipping from
    """
    try:
        # Load audio file
        audio = AudioSegment.from_file(str(audio_path))
        
        # Convert sample number to milliseconds
        # Samples are at SAMPLING_RATE (16000 Hz)
        start_ms = int((clip_start_samples / SAMPLING_RATE) * 1000)
        
        if start_ms >= len(audio):
            logger.error(f"Clip start sample {clip_start_samples} ({start_ms}ms) is beyond audio length {len(audio)}ms")
            return None
        
        # Extract audio from start_ms to the end
        clipped_audio = audio[start_ms:]
        
        # Generate output filename with sample number
        audio_stem = audio_path.stem
        output_filename = f"clip_{clip_start_samples}.mp3"
        output_path = config.PLAYGROUND_DIR / output_filename
        
        # Save clipped audio
        clipped_audio.export(str(output_path), format="mp3", bitrate="128k", parameters=["-q:a", "2"])
        
        logger.info(f"Saved clipped audio: {output_path} (from sample {clip_start_samples})")
        print(f"Saved clipped audio: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to clip audio file: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="moonshine",
        description="Live captioning from KQED stream using Moonshine models",
    )
    parser.add_argument(
        "--model_name",
        help="Model to run the demo with",
        default="moonshine/tiny",
        choices=["moonshine/base", "moonshine/tiny"],
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=None,
        help="Sample number to start clipping from (saves clipped audio, requires --file or uses most recent file in playground)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Audio file to process (relative to playground folder, or use most recent if not specified)",
    )
    args = parser.parse_args()
    model_name = args.model_name
    clip_start_samples = args.clip
    audio_file = args.file
    
    # If --clip is provided, save clipped audio file from playground folder
    if clip_start_samples is not None:
        # Find audio file in playground folder
        if audio_file:
            audio_path = config.PLAYGROUND_DIR / audio_file
        else:
            # Find most recent file starting with "moonshine" in playground
            audio_files = sorted(
                config.PLAYGROUND_DIR.glob("moonshine*.mp3"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if not audio_files:
                logger.error(f"No files starting with 'moonshine' found in playground folder: {config.PLAYGROUND_DIR}")
                exit(1)
            audio_path = audio_files[0]
        
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            exit(1)
        
        print(f"Clipping audio file: {audio_path}")
        print(f"Starting from sample: {clip_start_samples}")
        
        save_clipped_audio(audio_path, clip_start_samples)
    
    else:
        # Process stream as normal
        print(f"Loading Moonshine model '{model_name}' (using ONNX runtime) ...")
        transcribe = Transcriber(model_name=model_name, rate=SAMPLING_RATE)

        vad_model = load_silero_vad(onnx=True)
        vad_iterator = VADIterator(
            model=vad_model,
            sampling_rate=SAMPLING_RATE,
            threshold=config.VAD_THRESHOLD,
            min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
        )
        
        pcm_queue = Queue()
        shutdown_event = threading.Event()
        
        def signal_handler(signum, frame):
            logger.info(f"Signal {signum} received. Initiating graceful shutdown...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        ffmpeg_proc = run_ffmpeg(pcm_queue, shutdown_event)
        
        lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
        speech_buffer = []
        recording = False
        total_samples_processed = 0
        segment_start_samples = 0
        last_refresh_time = time.time()
        lookback_buffer = deque(maxlen=LOOKBACK_CHUNKS)
        
        # Buffer to hold raw PCM bytes for saving on shutdown
        pcm_archive_buffer = bytearray()

        print(f"Connected to stream: {config.STREAM_URL}")
        print("Press Ctrl+C to quit.\n")

        try:
            with requests.get(config.STREAM_URL, stream=True, timeout=20) as r:
                r.raise_for_status()

                for chunk in r.iter_content(chunk_size=4096):
                    if shutdown_event.is_set():
                        break
                    if not chunk:
                        continue
                    
                    # Feed raw stream to decoder
                    try:
                        ffmpeg_proc.stdin.write(chunk)
                        ffmpeg_proc.stdin.flush()
                    except BrokenPipeError:
                        break

                    while not pcm_queue.empty():
                        pcm_bytes = pcm_queue.get()
                        # Accumulate PCM bytes for saving
                        pcm_archive_buffer.extend(pcm_bytes)
                        
                        # Convert bytes back to float32 for VAD/Transcriber
                        pcm_chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        num_samples = len(pcm_chunk)
                        
                        # Track sample count
                        total_samples_processed += num_samples
                        
                        # Convert to torch tensor for Silero VAD
                        pcm_tensor = torch.from_numpy(pcm_chunk)
                        speech_dict = vad_iterator(pcm_tensor)
                        
                        if speech_dict:
                            # Start of speech detection
                            if "start" in speech_dict and not recording:
                                recording = True
                                lookback_samples = sum(len(c) for c in lookback_buffer)
                                segment_start_samples = total_samples_processed - lookback_samples
                                speech_buffer = list(lookback_buffer)
                                last_refresh_time = time.time()
                            
                            # End of speech detection
                            if "end" in speech_dict and recording:
                                full_speech = np.concatenate(speech_buffer)
                                buffer_samples = sum(len(c) for c in speech_buffer)
                                segment_end_samples = segment_start_samples + buffer_samples
                                text = transcribe(full_speech)
                                if text.strip():
                                    print(f"{segment_end_samples} {text}")
                                speech_buffer = []
                                vad_iterator.reset_states()
                                recording = False
                        
                        if recording:
                            speech_buffer.append(pcm_chunk)
                            
                            # Check for max duration
                            if (sum(len(c) for c in speech_buffer) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                                full_speech = np.concatenate(speech_buffer)
                                buffer_samples = sum(len(c) for c in speech_buffer)
                                segment_end_samples = segment_start_samples + buffer_samples
                                text = transcribe(full_speech)
                                if text.strip():
                                    print(f"{segment_end_samples} {text}")
                                speech_buffer = []
                                vad_iterator.reset_states()
                                recording = False
                            
                            # Periodic inference during recording
                            elif (time.time() - last_refresh_time) > MIN_REFRESH_SECS:
                                full_speech = np.concatenate(speech_buffer)
                                if len(full_speech) >= 2000:  # Minimum for Moonshine
                                    buffer_samples = sum(len(c) for c in speech_buffer)
                                    segment_end_samples = segment_start_samples + buffer_samples
                                    text = transcribe(full_speech)
                                    if text.strip():
                                        print(f"{segment_end_samples} {text}")
                                last_refresh_time = time.time()

                        if not recording:
                            # Update lookback buffer only when not in an active speech segment
                            lookback_buffer.append(pcm_chunk)

        except Exception as e:
            if not shutdown_event.is_set():
                logger.error(f"Error in stream processing: {e}")
        finally:
            shutdown_event.set()
            
            # Save audio on shutdown
            if pcm_archive_buffer:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"moonshine_{timestamp}.mp3"
                output_path = config.PLAYGROUND_DIR / filename
                config.PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
                
                print(f"\nSaving audio to {output_path}...")
                # Use ffmpeg to encode PCM to MP3
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "s16le",
                    "-ar", str(SAMPLING_RATE),
                    "-ac", "1",
                    "-i", "pipe:0",
                    "-f", "mp3",
                    "-b:a", "64k",
                    str(output_path)
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                proc.communicate(input=bytes(pcm_archive_buffer))
                file_size_mb = len(pcm_archive_buffer) / (1024 * 1024)
                logger.info(f"Saved audio: {output_path} ({file_size_mb:.2f} MB PCM)")
            
            if ffmpeg_proc:
                ffmpeg_proc.terminate()
            
            print(f"\nModel: {model_name}")
            print(f"Number of inferences: {transcribe.number_inferences}")
            if transcribe.number_inferences > 0:
                print(f"Mean inference time: {(transcribe.inference_secs / transcribe.number_inferences):.2f}s")
                print(f"Model realtime factor: {(transcribe.speech_secs / transcribe.inference_secs):.2f}x")