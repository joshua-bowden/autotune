import os
import sys
import json
from datetime import datetime
from pathlib import Path

import config
from search import extract_audio_segment
from utils import setup_logging, Transcriber, load_audio_file_to_numpy

logger = setup_logging(__name__)

# --- CONFIGURATION ---
# Set to None to process all lines, or a number to limit the test.
LIMIT = 20 
OUTPUT_DIR = config.DATA_DIR / "timing_test"
TRANSCRIPT_FILE = config.TRANSCRIPT_DIR / "current_transcript.txt"
# ---------------------

def parse_transcript_line(line, active_session=None):
    """
    Parses a line from current_transcript.txt:
    START_SAMPLES|END_SAMPLES|text
    Or a session marker: # SESSION YYYYMMDD_HHMMSS
    Returns (session_id, None, text, start_samples, end_samples)
    """
    line = line.strip()
    if line.startswith("# SESSION "):
        session_id = line.replace("# SESSION ", "").strip()
        return session_id, "MARKER", None, None, None
        
    parts = line.split('|')
    if len(parts) < 3:
        return None
    
    try:
        start_samples = int(parts[0])
        end_samples = int(parts[1])
        text = "|".join(parts[2:])
        return active_session, None, text, start_samples, end_samples
    except Exception as e:
        logger.error(f"Error parsing line: {line}. Error: {e}")
        return None

def transcribe_audio_clip(audio_path, transcriber):
    """
    Load an audio clip and transcribe it using Moonshine.
    Returns the transcription text.
    """
    samples = load_audio_file_to_numpy(audio_path)
    if samples is None:
        return None
    
    return transcriber(samples)

def main():
    if not TRANSCRIPT_FILE.exists():
        print(f"Transcript file not found: {TRANSCRIPT_FILE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Moonshine transcriber
    try:
        transcriber = Transcriber(model_name="moonshine/tiny")
    except Exception as e:
        logger.error(f"Failed to initialize Moonshine: {e}")
        transcriber = None
    
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if LIMIT:
        lines = lines[:LIMIT]

    active_session = None
    for i, line in enumerate(lines):
        parsed = parse_transcript_line(line, active_session)
        if not parsed:
            continue
            
        session_id, marker, text, start_samples, end_samples = parsed
        
        if marker == "MARKER":
            active_session = session_id
            continue
        
        # Format a name for the test clip
        query_name = f"test_{i+1:03d}_s{start_samples}"
        
        # Extract audio segment
        clip_path = extract_audio_segment(i, query=query_name, start_samples=start_samples, end_samples=end_samples, session_id=session_id)
        
        if clip_path:
            # Move from RESULTS_DIR to our specific timing_test dir
            target_path = OUTPUT_DIR / Path(clip_path).name
            if os.path.exists(clip_path):
                import shutil
                shutil.move(clip_path, target_path)
                
                # Transcribe with Moonshine
                moonshine_text = None
                if transcriber is not None:
                    moonshine_text = transcribe_audio_clip(target_path, transcriber)
                
                # Output: prettier format
                print(f"{i+1:2d}. Original: {text}")
                if moonshine_text:
                    print(f"    Moonshine: {moonshine_text}")
                else:
                    print(f"    Moonshine: [transcription failed]")
                print()
            else:
                print(f"{i+1:2d}. Original: {text}")
                print(f"    Moonshine: [file not found]")
                print()
        else:
            print(f"{i+1:2d}. Original: {text}")
            print(f"    Moonshine: [extract failed]")
            print()

if __name__ == "__main__":
    main()
