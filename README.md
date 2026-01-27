# Radio Archiver (Live Mode)

A professional radio stream archiving system that captures live audio, transcribes it using AI (Moonshine), segments content into stories using Google Gemini, and enables sample-accurate semantic search over the archive.

## Features

- **Sample-Perfect Timing**: Uses exact sample offsets instead of wall-clock time to eliminate audio drift.
- **Live AI Transcription**: Real-time transcription using the Moonshine speech-to-text model.
- **Story Segmentation**: Uses Google Gemini AI to identify and segment distinct stories from the transcript.
- **Semantic Search**: Vector-based search (TiDB Vector) to find stories by meaning.
- **Deterministic Archiving**: Rolling 1-minute MP3 archives aligned perfectly to sample boundaries.

## Architecture

The system consists of two main services managed by an orchestrator:

1. **live_transcriber.py**: Captures the stream, performs VAD (Voice Activity Detection), transcribes speech, and saves sample-aligned MP3 archives.
2. **processor.py**: Monitors the transcript, batches segments, and uses Gemini to identify stories, generate summaries, and create embeddings.
3. **search.py**: CLI for performing semantic searches and extracting sample-perfect audio clips.

All services are managed together using **run_all.py**.

## Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**
   Create a `.env` file with your credentials:
   ```
   TIDB_HOST=your_tidb_host
   TIDB_PORT=4000
   TIDB_USER=your_username
   TIDB_PASSWORD=your_password
   TIDB_DB_NAME=your_database
   GOOGLE_API_KEY=your_gemini_api_key
   STREAM_URL=https://streams.kqed.org/
   ```

3. **Initialize the database**
   ```bash
   python database.py
   ```

## Usage

### Running the System
Start the capture and processing pipeline:
```bash
python run_all.py
```

### Searching the Archive
Search for stories using natural language:
```bash
python search.py climate change policy
```
The search utility will automatically extract the exact audio clip for each result using sample offsets.

## Data Structure

- **Transcript Format**: `YYMMDD_HHMMSS|START_SAMPLES|END_SAMPLES|text`
- **Archive Filename**: `kqed_YYYYMMDD_HHMMSS_OFFSET.mp3`

## License
Educational and personal use.
