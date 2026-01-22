"""
Cleanup script to reset the local data environment and clear the database.
"""
import shutil
import os
from pathlib import Path
import config
import database
from utils import setup_logging

logger = setup_logging(__name__)

def reset_environment():
    # 1. Clear database table
    try:
        logger.info("Connecting to database to clear 'radio_stories' table...")
        with database.get_connection() as conn:
            cursor = conn.cursor()
            # TRUNCATE is faster and resets auto-increment IDs
            cursor.execute("TRUNCATE TABLE radio_stories")
            conn.commit()
            cursor.close()
            logger.info("Database table 'radio_stories' cleared.")
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")

    # 2. Delete data folder
    data_dir = config.DATA_DIR
    if data_dir.exists():
        try:
            logger.info(f"Deleting data directory: {data_dir}")
            shutil.rmtree(data_dir)
            logger.info("Data directory deleted.")
        except Exception as e:
            logger.error(f"Failed to delete data directory: {e}")
    else:
        logger.info("Data directory does not exist, skipping deletion.")

    # 3. Re-initialize directories (empty)
    logger.info("Re-initializing empty data directories...")
    config.initialize_directories()
    logger.info("Environment reset complete.")

if __name__ == "__main__":
    confirm = input("Are you sure you want to delete ALL data and clear the database? (y/n): ")
    if confirm.lower() == 'y':
        reset_environment()
    else:
        print("Reset cancelled.")
