"""
Personalization script: Finds the two semantically farthest stories and concatenates them.
"""
import json
import logging
from pathlib import Path
from pydub import AudioSegment
import database
import config
from search import extract_audio_segment
from utils import setup_logging

logger = setup_logging(__name__)

def find_farthest_stories():
    """Finds the pair of stories with the maximum cosine distance."""
    sql = """
    SELECT s1.id as id1, s2.id as id2, 
           s1.summary as summary1, s2.summary as summary2,
           s1.transcript as transcript1, s2.transcript as transcript2,
           s1.audio_path as audio_path1, s2.audio_path as audio_path2,
           vec_cosine_distance(s1.embedding, s2.embedding) as distance
    FROM radio_stories s1, radio_stories s2
    WHERE s1.id < s2.id
    ORDER BY distance DESC
    LIMIT 1
    """
    try:
        with database.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql)
            result = cursor.fetchone()
            cursor.close()
            return result
    except Exception as e:
        logger.error(f"Error finding farthest stories: {e}")
        return None

def create_personalization():
    logger.info("Finding the two most different stories in the database...")
    pair = find_farthest_stories()
    
    if not pair:
        logger.warning("Could not find enough stories to compare. Need at least 2.")
        return

    logger.info(f"Farthest pair found: (ID {pair['id1']}) and (ID {pair['id2']})")
    logger.info(f"Semantic distance: {pair['distance']:.4f}")

    # 1. Extract audio for both
    clips = []
    for i in [1, 2]:
        story_id = pair[f'id{i}']
        summary = pair[f'summary{i}']
        audio_meta = json.loads(pair[f'audio_path{i}'])
        
        logger.info(f"Extracting clip for story {story_id}: {summary[:50]}...")
        clip_path = extract_audio_segment(
            story_id=story_id,
            query=f"dist_story_{i}_{story_id}",
            start_samples=audio_meta.get("start_samples"),
            end_samples=audio_meta.get("end_samples"),
            session_id=audio_meta.get("session_id"),
        )
        if clip_path:
            clips.append(clip_path)

    if len(clips) < 2:
        logger.error("Failed to extract both audio clips.")
        return

    # 2. Concatenate with beep
    logger.info(f"Concatenating clips with {config.PERS_BEEP_DURATION_MS}ms beep gap...")
    from pydub.generators import Sine
    audio1 = AudioSegment.from_mp3(clips[0])
    audio2 = AudioSegment.from_mp3(clips[1])
    
    # Generate beep from config
    beep = Sine(config.PERS_BEEP_FREQ_HZ).to_audio_segment(
        duration=config.PERS_BEEP_DURATION_MS
    ).apply_gain(config.PERS_BEEP_GAIN_DB)
    
    combined = audio1 + beep + audio2
    
    # 3. Setup output directory and filename
    from datetime import datetime
    pers_dir = config.PERS_DIR
    pers_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%d_%m_%Y_%H%M") 
    base_filename = f"personalization_story{pair['id1']}+{pair['id2']}"
    
    output_audio = pers_dir / f"{base_filename}.mp3"
    combined.export(str(output_audio), format="mp3")
    logger.info(f"Audio mashup saved to: {output_audio}")

    # 4. Create text report
    output_text = pers_dir / f"{base_filename}.txt"
    
    # Calculate transition times
    switch_start = len(audio1) / 1000.0
    switch_end = (len(audio1) + len(beep)) / 1000.0
    
    with open(output_text, "w", encoding="utf-8") as f:
        f.write("=== SEMANTICALLY FARTHEST STORIES ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Pair Distance: {pair['distance']:.4f}\n")
        f.write(f"Transition Beep: {switch_start:.2f}s - {switch_end:.2f}s\n\n")
        
        for i in [1, 2]:
            marker = "START" if i == 1 else f"after {switch_end:.2f}s"
            f.write(f"--- STORY {i} (ID: {pair[f'id{i}']} | Position: {marker}) ---\n")
            f.write(f"SUMMARY: {pair[f'summary{i}']}\n")
            
            # Parse text from transcript blob if it's JSON
            try:
                story_data = json.loads(pair[f'transcript{i}'])
                full_text = story_data.get("text", pair[f'transcript{i}'])
            except:
                full_text = pair[f'transcript{i}']
                
            f.write(f"TEXT: {full_text.strip()}\n\n")

    logger.info(f"Report saved to: {output_text}")
    print(f"\nCreated personalization result!")
    print(f"Audio: {output_audio}")
    print(f"Report: {output_text}")

if __name__ == "__main__":
    create_personalization()
