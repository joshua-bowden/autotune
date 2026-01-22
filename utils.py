"""
Shared utility functions for the radio archiver system.

This module provides common functionality used across multiple components:
- Embedding generation using Google's Gemini API
- Rate limiting and usage tracking
- Logging configuration
"""

import logging
import time
from typing import Optional, List

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
