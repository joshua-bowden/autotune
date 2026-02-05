"""
Personalization script: Finds N semantically farthest stories and concatenates them.
Uses greedy k-dispersion with cosine distance so the set is maximally diverse.
"""
import argparse
import json
import math
from pathlib import Path
from datetime import datetime

from pydub import AudioSegment

import database
import config
from search import extract_audio_segment
from utils import setup_logging

logger = setup_logging(__name__)


def _l2_norm(v):
    return math.sqrt(sum(x * x for x in v))


def _normalize(v):
    n = _l2_norm(v)
    if n <= 0:
        return v
    return [x / n for x in v]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cosine_distance(u, v):
    """Cosine distance = 1 - cosine_similarity. For normalized u,v this is 1 - dot(u,v)."""
    return 1.0 - _dot(u, v)


def select_farthest_k(stories_with_embeddings, k):
    """
    Select k stories that are maximally far from each other using greedy k-dispersion.

    Uses cosine distance (1 - cos_sim). At each step we add the story whose
    minimum distance to the already-selected set is largest.

    Returns:
        List of k story dicts (same shape as input items) in selection order.
    """
    n = len(stories_with_embeddings)
    if n < k:
        logger.warning(f"Only {n} stories available; requested k={k}. Using all.")
        k = n
    if k <= 0:
        return []

    # Normalize embeddings so cosine_sim(a,b) = dot(a,b), cosine_distance = 1 - dot
    normalized = [_normalize(s["embedding"]) for s in stories_with_embeddings]

    def cos_dist(i, j):
        return _cosine_distance(normalized[i], normalized[j])

    # Greedy: start with the pair with maximum cosine distance
    best_pair = (0, 1)
    best_dist = cos_dist(0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            d = cos_dist(i, j)
            if d > best_dist:
                best_dist = d
                best_pair = (i, j)

    selected_indices = list(best_pair)

    # Add remaining k-2: each time add the story farthest from the current set
    for _ in range(k - 2):
        best_p = None
        best_min_dist = -1.0
        for p in range(n):
            if p in selected_indices:
                continue
            min_dist_to_set = min(cos_dist(p, s) for s in selected_indices)
            if min_dist_to_set > best_min_dist:
                best_min_dist = min_dist_to_set
                best_p = p
        selected_indices.append(best_p)

    return [stories_with_embeddings[i] for i in selected_indices]


def create_personalization(k: int = 2):
    logger.info(f"Finding the {k} most different stories (greedy k-dispersion)...")

    stories = database.get_all_stories_with_embeddings()
    if not stories:
        logger.warning("No stories in database.")
        return
    if len(stories) < 2:
        logger.warning("Need at least 2 stories.")
        return

    selected = select_farthest_k(stories, k)
    ids = [s["id"] for s in selected]
    logger.info(f"Selected story IDs (in order): {ids}")

    # Log pairwise distances for the selected set
    normalized = [_normalize(s["embedding"]) for s in selected]
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            d = _cosine_distance(normalized[i], normalized[j])
            logger.info(f"  Distance ID {selected[i]['id']} <-> ID {selected[j]['id']}: {d:.4f}")

    # 1. Extract audio for each
    clips = []
    for i, story in enumerate(selected):
        story_id = story["id"]
        summary = story["summary"]
        audio_meta = json.loads(story["audio_path"])

        logger.info(f"Extracting clip for story {story_id}: {summary[:50]}...")
        clip_path = extract_audio_segment(
            story_id=story_id,
            query=f"personalization_story_{i + 1}_{story_id}",
            start_samples=audio_meta.get("start_samples"),
            end_samples=audio_meta.get("end_samples"),
            session_id=audio_meta.get("session_id"),
        )
        if clip_path:
            clips.append(clip_path)

    if len(clips) < len(selected):
        logger.error("Failed to extract some audio clips.")
        return

    # 2. Concatenate with beeps between
    logger.info(f"Concatenating {len(clips)} clips with {config.PERS_BEEP_DURATION_MS}ms beep gaps...")
    from pydub.generators import Sine
    beep = Sine(config.PERS_BEEP_FREQ_HZ).to_audio_segment(
        duration=config.PERS_BEEP_DURATION_MS
    ).apply_gain(config.PERS_BEEP_GAIN_DB)

    combined = AudioSegment.empty()
    segment_lengths_ms = []
    for i, clip_path in enumerate(clips):
        seg = AudioSegment.from_mp3(clip_path)
        segment_lengths_ms.append(len(seg))
        if i > 0:
            combined += beep
        combined += seg

    # 3. Output directory and filename
    pers_dir = config.PERS_DIR
    pers_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%d_%m_%Y_%H%M")
    base_filename = f"personalization_{'+'.join(str(s['id']) for s in selected)}"

    output_audio = pers_dir / f"{base_filename}.mp3"
    combined.export(str(output_audio), format="mp3")
    logger.info(f"Audio mashup saved to: {output_audio}")

    # 4. Text report (use same segment lengths as combined audio)
    output_text = pers_dir / f"{base_filename}.txt"
    with open(output_text, "w", encoding="utf-8") as f:
        f.write("=== SEMANTICALLY FARTHEST STORIES (GREEDY K-DISPERSION) ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Number of stories: {k}\n")
        f.write(f"Story IDs (order): {ids}\n\n")

        elapsed_ms = 0
        for i, story in enumerate(selected):
            marker = "START" if i == 0 else f"after {elapsed_ms / 1000.0:.2f}s (beep)"
            f.write(f"--- STORY {i + 1} (ID: {story['id']} | Position: {marker}) ---\n")
            f.write(f"SUMMARY: {story['summary']}\n")
            try:
                story_data = json.loads(story["transcript"])
                full_text = story_data.get("text", story["transcript"])
            except Exception:
                full_text = story["transcript"]
            f.write(f"TEXT: {full_text.strip()}\n\n")

            elapsed_ms += segment_lengths_ms[i]
            if i < len(clips) - 1:
                elapsed_ms += config.PERS_BEEP_DURATION_MS

    logger.info(f"Report saved to: {output_text}")
    print(f"\nCreated personalization result!")
    print(f"Audio: {output_audio}")
    print(f"Report: {output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find N semantically farthest stories and create a combined audio + report."
    )
    parser.add_argument(
        "num",
        type=int,
        nargs="?",
        default=2,
        help="Number of stories to select (default: 2)",
    )
    args = parser.parse_args()
    if args.num < 2:
        parser.error("num must be at least 2")
    create_personalization(k=args.num)
