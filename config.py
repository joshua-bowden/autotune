"""
Configuration module for the radio archiver system.

This module loads configuration from environment variables for security.
Sensitive credentials should never be hardcoded in source code.

Required environment variables:
    - TIDB_HOST: TiDB database host
    - TIDB_PORT: TiDB database port
    - TIDB_USER: TiDB username
    - TIDB_PASSWORD: TiDB password
    - TIDB_DB_NAME: TiDB database name
    - GOOGLE_API_KEY: Google Gemini API key
    - STREAM_URL: Radio stream URL
    - CHUNK_DURATION: Audio chunk duration in seconds
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
    """
    Get environment variable or exit with error if required and missing.
    
    Args:
        var_name: Name of the environment variable
        default: Default value if variable is not set (None means required)
        
    Returns:
        Value of the environment variable
        
    Raises:
        SystemExit: If required variable is missing
    """
    value = os.getenv(var_name, default)
    if value is None:
        print(f"ERROR: Required environment variable '{var_name}' is not set.")
        print("Please create a .env file based on .env.example")
        sys.exit(1)
    return value


# TiDB Configuration
TIDB_HOST: str = get_env_variable("TIDB_HOST")
TIDB_PORT: int = int(get_env_variable("TIDB_PORT", "4000"))
TIDB_USER: str = get_env_variable("TIDB_USER")
TIDB_PASSWORD: str = get_env_variable("TIDB_PASSWORD")
TIDB_DB_NAME: str = get_env_variable("TIDB_DB_NAME", "test")

# Determine the absolute path to the TiDB certificate
BASE_DIR: Path = Path(__file__).parent
TIDB_CA_PATH: str = str(BASE_DIR / "tidb.pem")

# Gemini Configuration
GOOGLE_API_KEY: str = get_env_variable("GOOGLE_API_KEY")
EMBEDDING_DIMENSIONS: int = 768  # text-embedding-004 dimension

# Application - Core
STREAM_URL: str = "https://streams.kqed.org/"

# VAD (Voice Activity Detection) - Controls speech/silence detection
VAD_SAMPLING_RATE: int = 16000  # Samples per second (16kHz required by Silero/Moonshine)
VAD_CHUNK_SIZE: int = 512       # Samples per processing step (256, 512, or 1024 for Silero)
VAD_MIN_SILENCE_MS: int = 300    # Wait this long after someone stops talking to end the segment
VAD_THRESHOLD: float = 0.5       # Speech probability threshold (0.0 to 1.0)

# Engine - Archiving & Transcription
# Number of chunks to keep in memory before speech is detected.
# When a "start" event happens, we prepend these chunks to capture leading consonants/sounds.
LOOKBACK_CHUNKS: int = 40 

# Live Mode settings
LIVE_MIN_REFRESH_S: float = 0.2
LIVE_MAX_SPEECH_S: float = 15.0
# Duration of continuous VAD silence before annotating transcript
SILENCE_ANNOUNCE_S: float = 3.0

# Frame-aligned archival: 16kHz * 60s = 960,000 samples. 
# 960,000 is exactly 1875 chunks of 512 samples.
# Ensures file-to-file contiguity without losing or repeating samples.
SAMPLES_PER_ARCHIVE: int = 960000 

# Story Processing (processor.py)
WINDOW_SIZE_SENTENCES: int = 100
OVERLAP_SENTENCES: int = 0
MIN_STORY_LENGTH: int = 10  # Minimum characters for a valid story
MONITOR_INTERVAL_S: int = 5  # Seconds between directory scans
GEMINI_STORY_MODEL: str = "gemini-2.5-flash-lite"
LLM_RATE_LIMIT_DELAY_S: int = 12 # Seconds to wait between Gemini calls

# Search & Clipping (search.py)
SEARCH_DEFAULT_TOP_K: int = 2
SEARCH_EXCERPT_LENGTH: int = 100
CLIP_PADDING_MS: int = 0

# Personalization (personalization.py)
PERS_BEEP_FREQ_HZ: int = 1000
PERS_BEEP_DURATION_MS: int = 1000
PERS_BEEP_GAIN_DB: int = -20

# Directory Configuration
DATA_DIR: Path = BASE_DIR / "data"
AUDIO_DIR: Path = DATA_DIR / "audio"
TRANSCRIPT_DIR: Path = DATA_DIR / "transcripts"
STORY_DIR: Path = DATA_DIR / "stories"
RESULTS_DIR: Path = DATA_DIR / "search"
PERS_DIR: Path = DATA_DIR / "personalization"
PLAYGROUND_DIR: Path = DATA_DIR / "playground"


def initialize_directories() -> None:
    """Create required data directories if they don't exist."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PERS_DIR.mkdir(parents=True, exist_ok=True)
    PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)


# Initialize directories on module import
initialize_directories()
