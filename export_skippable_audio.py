"""
Export one combined audio clip per story from stories_log.jsonl.

Reads data/stories/stories_log.jsonl, and for each story concatenates the
audio for all matching_segments (using sample offsets and session_id from
data/audio), then saves one MP3 per story to data/skippable.

Usage:
    python export_skippable_audio.py [--limit N]

Example:
    python export_skippable_audio.py
    python export_skippable_audio.py --limit 10
"""

import argparse
import json
import re
from pathlib import Path

from pydub import AudioSegment

import config
from search import get_audio_segment
from utils import setup_logging

logger = setup_logging(__name__)

STORIES_LOG = config.STORY_DIR / "stories_log.jsonl"
SKIPPABLE_DIR = config.DATA_DIR / "skippable"


def safe_filename(index: int, summary: str, max_len: int = 60) -> str:
    """Build a filesystem-safe base name for the clip (no extension)."""
    safe = re.sub(r'[^\w\s-]', '', (summary or "story").strip()).replace(" ", "_")
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("_")
    if not safe:
        safe = "story"
    return f"{index:04d}_{safe}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one combined audio clip per story to data/skippable.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N stories (default: all)")
    args = parser.parse_args()

    if not STORIES_LOG.exists():
        logger.error("Stories log not found: %s", STORIES_LOG)
        raise SystemExit(1)

    SKIPPABLE_DIR.mkdir(parents=True, exist_ok=True)

    # File is pretty-printed JSON with "---" record separators between objects.
    # Parse by decoding one object at a time with JSONDecoder.raw_decode.
    stories = []
    with open(STORIES_LOG, "r", encoding="utf-8") as f:
        content = f.read()
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        # Skip whitespace and commas between objects
        while pos < len(content) and content[pos] in " \t\n\r,":
            pos += 1
        if pos >= len(content):
            break
        # Skip "---" record separator lines (not valid JSON)
        if content[pos : pos + 3] == "---":
            pos += 3
            continue
        try:
            obj, end = decoder.raw_decode(content, pos)
            stories.append(obj)
            pos = end
        except json.JSONDecodeError as e:
            logger.warning("Skip invalid JSON at position %d: %s", pos, e)
            pos += 1

    if args.limit is not None:
        stories = stories[: args.limit]

    logger.info("Processing %d stories from %s -> %s", len(stories), STORIES_LOG, SKIPPABLE_DIR)

    for i, story in enumerate(stories):
        segments = story.get("matching_segments") or []
        if not segments:
            logger.warning("Story %d has no matching_segments, skipping.", i)
            continue

        combined = AudioSegment.empty()
        failed = False
        for seg in segments:
            start_samples = seg.get("start_samples")
            end_samples = seg.get("end_samples")
            session_id = seg.get("session_id")
            if start_samples is None or end_samples is None:
                logger.warning("Story %d segment missing start_samples/end_samples, skipping segment.", i)
                failed = True
                break
            segment_audio = get_audio_segment(
                start_samples=start_samples,
                end_samples=end_samples,
                session_id=session_id,
            )
            if segment_audio is None:
                logger.warning("Story %d: could not get audio for segment %s-%s.", i, start_samples, end_samples)
                failed = True
                break
            combined += segment_audio

        if failed or len(combined) == 0:
            continue

        base_name = safe_filename(i, story.get("summary", ""))
        out_path = SKIPPABLE_DIR / f"{base_name}.mp3"
        combined.export(
            str(out_path),
            format="mp3",
            bitrate="128k",
            parameters=["-q:a", "2"],
        )
        logger.info("Wrote %s (%.1fs)", out_path.name, len(combined) / 1000.0)

    print(f"Done. Clips saved to {SKIPPABLE_DIR}")


if __name__ == "__main__":
    main()
