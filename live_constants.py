"""Constants for live transcription."""

import os
from pathlib import Path
import config

# Audio settings
SAMPLING_RATE = 16000
CHUNK_SIZE = 512  # Silero VAD requirement
LOOKBACK_CHUNKS = 5

# Transcription settings
MAX_SPEECH_SECS = 15
MIN_REFRESH_SECS = 0.2
MAX_LINE_LENGTH = 80

# File paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
TRANSCRIPT_PATH = TRANSCRIPT_DIR / "current_transcript.txt"
DRAFT_PATH = TRANSCRIPT_DIR / "live_draft.txt"

# Model names
MOONSHINE_MODEL = "moonshine/tiny" # Default to tiny for speed
VAD_MODEL_ONNX = True 

# Stream settings
STREAM_URL = config.STREAM_URL
