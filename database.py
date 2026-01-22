"""
Database operations for the radio archiver system.

This module handles all interactions with the TiDB database, including:
- Connection management
- Schema initialization
- Story storage with vector embeddings
- Vector similarity search

The database uses TiDB's vector capabilities to enable semantic search
over radio story transcripts using cosine similarity.
"""

import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Generator

import mysql.connector
from mysql.connector import Error as MySQLError

import config
from utils import setup_logging


logger = setup_logging(__name__)


# Constants
EMBEDDING_DIMENSIONS = config.EMBEDDING_DIMENSIONS
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


@contextmanager
def get_connection() -> Generator[mysql.connector.MySQLConnection, None, None]:
    """
    Context manager for database connections with automatic cleanup.
    
    Yields:
        Active database connection
        
    Raises:
        MySQLError: If connection fails after retries
        
    Example:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM radio_stories")
    """
    conn = None
    try:
        conn = mysql.connector.connect(
            host=config.TIDB_HOST,
            port=config.TIDB_PORT,
            user=config.TIDB_USER,
            password=config.TIDB_PASSWORD,
            database=config.TIDB_DB_NAME,
            ssl_ca=config.TIDB_CA_PATH,
            ssl_verify_cert=True
        )
        yield conn
    except MySQLError as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()


def init_db() -> bool:
    """
    Initialize the database schema.
    
    Creates the radio_stories table with vector support if it doesn't exist.
    The table stores radio story metadata along with 768-dimensional embeddings
    for semantic search.
    
    Returns:
        True if initialization successful, False otherwise
    """
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS radio_stories (
        id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME,
        start_time FLOAT,
        end_time FLOAT,
        transcript TEXT,
        summary TEXT,
        audio_path TEXT,
        embedding VECTOR({EMBEDDING_DIMENSIONS}),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_timestamp (timestamp)
    );
    """
    
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_query)
            conn.commit()
            cursor.close()
            logger.info("Database initialized successfully - table 'radio_stories' ready")
            return True
    except MySQLError as e:
        logger.error(f"Error initializing database: {e}")
        return False


def save_story(
    timestamp: Optional[str],
    start_time: float,
    end_time: float,
    transcript: str,
    summary: str,
    audio_path: str,
    embedding: List[float]
) -> Optional[int]:
    """
    Save a processed story to the database.
    
    Args:
        timestamp: Story timestamp (datetime string or None)
        start_time: Start time in seconds within audio file
        end_time: End time in seconds within audio file
        transcript: Full transcript text
        summary: Story summary/title
        audio_path: Path to audio file or JSON list of files
        embedding: 768-dimensional embedding vector
        
    Returns:
        ID of inserted story, or None if save failed
    """
    if len(embedding) != EMBEDDING_DIMENSIONS:
        logger.error(
            f"Invalid embedding dimension: expected {EMBEDDING_DIMENSIONS}, "
            f"got {len(embedding)}"
        )
        return None
    
    sql = """
    INSERT INTO radio_stories 
    (timestamp, start_time, end_time, transcript, summary, audio_path, embedding) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    # Convert embedding list to string format for TiDB Vector
    embedding_str = str(embedding)
    values = (timestamp, start_time, end_time, transcript, summary, audio_path, embedding_str)
    
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, values)
            conn.commit()
            story_id = cursor.lastrowid
            cursor.close()
            logger.info(f"Story saved to database (ID: {story_id})")
            return story_id
    except MySQLError as e:
        logger.error(f"Error saving story to database: {e}")
        return None


def search_stories(query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar stories using vector similarity.
    
    Uses TiDB's vec_cosine_distance function to find stories with embeddings
    most similar to the query embedding. Lower distance = higher similarity.
    
    Args:
        query_embedding: 768-dimensional query embedding vector
        top_k: Number of results to return (default: 5)
        
    Returns:
        List of story dictionaries with keys: id, timestamp, transcript,
        summary, audio_path, distance. Empty list if search fails.
    """
    if len(query_embedding) != EMBEDDING_DIMENSIONS:
        logger.error(
            f"Invalid query embedding dimension: expected {EMBEDDING_DIMENSIONS}, "
            f"got {len(query_embedding)}"
        )
        return []
    
    # Use parameterized query to prevent SQL injection
    sql = """
    SELECT id, timestamp, transcript, summary, audio_path, 
           vec_cosine_distance(embedding, %s) as distance
    FROM radio_stories
    ORDER BY distance ASC
    LIMIT %s
    """
    
    embedding_str = str(query_embedding)
    
    try:
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql, (embedding_str, top_k))
            results = cursor.fetchall()
            cursor.close()
            logger.info(f"Search returned {len(results)} results")
            return results
    except MySQLError as e:
        logger.error(f"Error searching stories: {e}")
        return []


def get_story_count() -> int:
    """
    Get the total number of stories in the database.
    
    Returns:
        Number of stories, or 0 if query fails
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM radio_stories")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
    except MySQLError as e:
        logger.error(f"Error getting story count: {e}")
        return 0


def parse_audio_metadata(audio_path_json: str) -> Dict[str, Any]:
    """
    Parse audio metadata from stored JSON string.
    
    Args:
        audio_path_json: JSON string from audio_path column
        
    Returns:
        Dictionary with parsed metadata, or empty dict if parsing fails
    """
    import json
    
    try:
        return json.loads(audio_path_json)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse audio metadata: {e}")
        return {}


if __name__ == "__main__":
    # Initialize database when run directly
    success = init_db()
    if success:
        count = get_story_count()
        logger.info(f"Database contains {count} stories")
