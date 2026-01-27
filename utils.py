"""
Shared utility functions for the radio archiver system.

This module provides common functionality used across multiple components:
- Embedding generation using Google's Gemini API
- Rate limiting and usage tracking
- Logging configuration
- Audio transcription using Moonshine
- Audio format conversion utilities
- Text normalization utilities
"""

import logging
import time
from typing import Optional, List
from pathlib import Path
import numpy as np
from pydub import AudioSegment

from google import genai
from google.genai import types

import config


# Constants
DAILY_REQUEST_LIMIT = 1500  # Conservative limit for free tier (1500/day)
RATE_LIMIT_WINDOW_SECONDS = 86400  # 24 hours
RATE_LIMIT_SLEEP_SECONDS = 60  # Wait time when limit reached


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class UsageTracker:
    """
    Tracks API usage to stay within daily rate limits.
    
    Implements a rolling 24-hour window to prevent exceeding the Gemini API
    free tier limit of 1500 requests per day.
    """
    
    def __init__(self, daily_limit: int = DAILY_REQUEST_LIMIT):
        """
        Initialize the usage tracker.
        
        Args:
            daily_limit: Maximum requests allowed per 24-hour period
        """
        self.daily_requests = 0
        self.last_reset = time.time()
        self.daily_limit = daily_limit
        self.logger = setup_logging(__name__)
    
    def check_and_increment(self, count: int = 1) -> bool:
        """
        Check if request can be made within rate limits and increment counter.
        
        Args:
            count: Number of requests to add (default: 1)
            
        Returns:
            True if request is allowed, False if limit reached
        """
        # Reset counter if 24 hours have passed
        if time.time() - self.last_reset > RATE_LIMIT_WINDOW_SECONDS:
            self.daily_requests = 0
            self.last_reset = time.time()
            self.logger.info("Rate limit counter reset")
        
        # Check if adding this request would exceed the limit
        if self.daily_requests + count > self.daily_limit:
            self.logger.warning(
                f"Daily rate limit reached ({self.daily_requests}/{self.daily_limit}). "
                f"Pausing for {RATE_LIMIT_SLEEP_SECONDS} seconds..."
            )
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)
            return False
        
        self.daily_requests += count
        return True


# Global usage tracker instance
_usage_tracker = UsageTracker()


def get_embedding(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
    model: str = "text-embedding-004"
) -> Optional[List[float]]:
    """
    Generate embedding vector for text using Gemini API.
    
    Args:
        text: Text to embed
        task_type: Embedding task type ("RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY")
        model: Embedding model to use
        
    Returns:
        List of embedding values, or None if generation fails
    """
    logger = setup_logging(__name__)
    
    # Check rate limits
    if not _usage_tracker.check_and_increment():
        logger.error("Embedding generation skipped due to rate limit")
        return None
    
    try:
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        response = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        return response.embeddings[0].values
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def get_batch_embeddings(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    model: str = "text-embedding-004"
) -> Optional[List[List[float]]]:
    """
    Generate embeddings for multiple texts in a single API call.
    
    This is more efficient than calling get_embedding() multiple times
    as it uses only one API request.
    
    Args:
        texts: List of texts to embed
        task_type: Embedding task type
        model: Embedding model to use
        
    Returns:
        List of embedding vectors, or None if generation fails
    """
    logger = setup_logging(__name__)
    
    if not texts:
        return []
    
    # Batch embedding still counts as 1 API request
    if not _usage_tracker.check_and_increment():
        logger.error("Batch embedding generation skipped due to rate limit")
        return None
    
    try:
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        response = client.models.embed_content(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        return [e.values for e in response.embeddings]
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return None


def reset_usage_tracker() -> None:
    """Reset the global usage tracker (useful for testing)."""
    global _usage_tracker
    _usage_tracker = UsageTracker()


# ============================================================================
# Text Normalization Utilities
# ============================================================================

def normalize_word(word: str) -> str:
    """
    Normalize word for matching (lowercase, alphanumeric only).
    
    Used for comparing words across different transcriptions where punctuation
    and capitalization may differ.
    
    Args:
        word: Word to normalize
        
    Returns:
        Normalized word (lowercase, alphanumeric characters only)
    """
    return "".join(c.lower() for c in word if c.isalnum())


# ============================================================================
# Audio Transcription Utilities
# ============================================================================

class Transcriber:
    """
    Wrapper for Moonshine ONNX speech-to-text model.
    
    Provides a consistent interface for transcribing audio samples using
    the Moonshine model with proper initialization and error handling.
    """
    
    def __init__(self, model_name: str = "moonshine/tiny"):
        """
        Initialize the Moonshine transcriber.
        
        Args:
            model_name: Moonshine model to use ("moonshine/tiny" or "moonshine/base")
        """
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.tokenizer = load_tokenizer()
        self.rate = config.VAD_SAMPLING_RATE
        
        # Warmup the model to avoid first-inference latency
        self.model.generate(np.zeros((1, int(self.rate)), dtype=np.float32))
    
    def __call__(self, speech: np.ndarray) -> str:
        """
        Transcribe audio samples to text.
        
        Args:
            speech: Audio samples as numpy array (float32, normalized to [-1, 1])
            
        Returns:
            Transcribed text string, or empty string if transcription fails
        """
        # Moonshine expects a minimum amount of audio to process through its convolutions
        if len(speech) < 2000:  # ~125ms minimum at 16kHz
            return ""
        
        try:
            tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
            text = self.tokenizer.decode_batch(tokens)[0]
            return text
        except Exception as e:
            logger = setup_logging(__name__)
            logger.error(f"Transcription error: {e}")
            return ""


# ============================================================================
# Audio Format Conversion Utilities
# ============================================================================

def audio_segment_to_numpy(audio: AudioSegment) -> np.ndarray:
    """
    Convert pydub AudioSegment to numpy array suitable for transcription.
    
    Converts audio to mono, ensures correct sample rate, and normalizes to float32
    in the range [-1, 1] as expected by Moonshine and other audio processing models.
    
    Args:
        audio: pydub AudioSegment to convert
        
    Returns:
        Numpy array of float32 samples normalized to [-1, 1]
    """
    # Convert to mono if needed
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Resample to VAD_SAMPLING_RATE (16kHz) if needed
    if audio.frame_rate != config.VAD_SAMPLING_RATE:
        audio = audio.set_frame_rate(config.VAD_SAMPLING_RATE)
    
    # Convert to numpy array (float32, normalized to [-1, 1])
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Normalize based on sample width
    if audio.sample_width == 2:  # 16-bit
        samples = samples / 32768.0
    elif audio.sample_width == 4:  # 32-bit
        samples = samples / 2147483648.0
    else:
        samples = samples / (2 ** (audio.sample_width * 8 - 1))
    
    return samples


def load_audio_file_to_numpy(audio_path: Path) -> Optional[np.ndarray]:
    """
    Load an audio file and convert it to numpy array for transcription.
    
    Convenience function that loads MP3, converts format, and returns numpy array.
    
    Args:
        audio_path: Path to audio file (MP3, WAV, etc.)
        
    Returns:
        Numpy array of float32 samples, or None if loading fails
    """
    logger = setup_logging(__name__)
    
    if not audio_path.exists():
        logger.error(f"Audio file does not exist: {audio_path}")
        return None
    
    # Check file size - empty or very small files are likely corrupted
    file_size = audio_path.stat().st_size
    if file_size < 100:  # MP3 files should be at least 100 bytes
        logger.error(f"Audio file appears corrupted (too small: {file_size} bytes): {audio_path}")
        return None
    
    try:
        # Try loading as MP3 first
        audio = AudioSegment.from_mp3(str(audio_path))
        return audio_segment_to_numpy(audio)
    except Exception as e:
        # If MP3 loading fails, try loading as generic audio format
        try:
            audio = AudioSegment.from_file(str(audio_path))
            return audio_segment_to_numpy(audio)
        except Exception as e2:
            logger.error(f"Error loading audio file {audio_path}: {e2}")
            return None
