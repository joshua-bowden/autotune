# Radio Archiver

A professional radio stream archiving system that captures live audio, transcribes it using AI, segments content into stories, and enables semantic search over the archive.

## Features

- **Live Audio Capture**: Continuously records audio from a radio stream in time-based chunks
- **AI Transcription**: Converts audio to text using the Moonshine speech-to-text model
- **Story Segmentation**: Uses Google Gemini AI to identify and segment distinct stories
- **Semantic Search**: Vector-based search to find stories by meaning, not just keywords
- **Scalable Storage**: Uses TiDB with vector support for efficient storage and retrieval

## Architecture

The system consists of four main components:

1. **capture.py** - Streams and saves audio chunks from the radio station
2. **transcribe.py** - Monitors audio files and generates transcripts
3. **processor.py** - Segments transcripts into stories and generates embeddings
4. **search.py** - Command-line interface for searching the archive

All services can be managed together using **run_all.py**.

## Prerequisites

- Python 3.8 or higher
- TiDB database with vector support
- Google Gemini API key
- FFmpeg (for audio processing with pydub)

## Installation

1. **Clone the repository**
   ```bash
   cd /home/josh/ecj
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and fill in your credentials:
   ```
   TIDB_HOST=your_tidb_host
   TIDB_PORT=4000
   TIDB_USER=your_username
   TIDB_PASSWORD=your_password
   TIDB_DB_NAME=your_database
   GOOGLE_API_KEY=your_gemini_api_key
   STREAM_URL=https://streams.kqed.org/
   CHUNK_DURATION=60
   ```

4. **Initialize the database**
   ```bash
   python database.py
   ```

## Usage

### Running All Services

Start all services together (recommended):

```bash
python run_all.py
```

This will start:
- Audio capture
- Transcription
- Story processing

Press `Ctrl+C` to stop all services gracefully.

### Running Individual Services

You can also run services independently:

```bash
# Capture audio only
python capture.py

# Transcribe audio only
python transcribe.py

# Process transcripts only
python processor.py
```

### Searching the Archive

Search for stories using natural language:

```bash
python search.py climate change policy
python search.py local election results
python search.py traffic updates
```

The search uses semantic similarity, so it finds stories by meaning rather than exact keyword matches.

## Configuration

All configuration is managed through environment variables in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `TIDB_HOST` | TiDB database host | Required |
| `TIDB_PORT` | TiDB database port | 4000 |
| `TIDB_USER` | Database username | Required |
| `TIDB_PASSWORD` | Database password | Required |
| `TIDB_DB_NAME` | Database name | test |
| `GOOGLE_API_KEY` | Gemini API key | Required |
| `STREAM_URL` | Radio stream URL | https://streams.kqed.org/ |
| `CHUNK_DURATION` | Audio chunk duration (seconds) | 60 |

## Rate Limiting Strategy

The system implements intelligent rate limiting for the Gemini API free tier (1500 requests/day):

- **Batch Processing**: Transcripts are batched (5 files = ~5 minutes) before processing
- **Batch Embeddings**: All story embeddings in a batch are generated in a single API call
- **Usage Tracking**: Monitors daily request count with automatic throttling
- **Conservative Limits**: Targets 1400 requests/day to stay safely under the limit

This approach allows processing approximately:
- 280 batches per day (5 minutes each)
- ~1400 minutes of radio per day (~23 hours)

## Directory Structure

```
ecj/
├── capture.py              # Audio capture service
├── transcribe.py           # Transcription service
├── processor.py            # Story processing service
├── search.py               # Search CLI
├── run_all.py              # Service orchestrator
├── database.py             # Database operations
├── config.py               # Configuration management
├── utils.py                # Shared utilities
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── tidb.pem                # TiDB SSL certificate
└── data/                   # Data directory
    ├── audio/              # Captured audio chunks
    ├── transcripts/        # Generated transcripts
    └── transcripts_processed/  # Processed transcripts
```

## Database Schema

The `radio_stories` table stores:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INT | Primary key |
| `timestamp` | DATETIME | Story timestamp |
| `start_time` | FLOAT | Start time in audio |
| `end_time` | FLOAT | End time in audio |
| `transcript` | TEXT | Full transcript |
| `summary` | TEXT | Story summary/title |
| `audio_path` | VARCHAR(255) | Path to audio file(s) |
| `embedding` | VECTOR(768) | Semantic embedding vector |
| `created_at` | TIMESTAMP | Record creation time |

## Troubleshooting

### "Required environment variable not set"
- Ensure you've created a `.env` file from `.env.example`
- Check that all required variables are filled in

### "Failed to load Moonshine model"
- Update transformers: `pip install --upgrade transformers`
- Ensure you have sufficient disk space for model download

### "Database connection error"
- Verify TiDB credentials in `.env`
- Check that `tidb.pem` certificate file exists
- Ensure network connectivity to TiDB host

### "Daily rate limit reached"
- The system will automatically pause when approaching limits
- Consider reducing `BATCH_SIZE` in `processor.py` for slower processing
- Upgrade to Gemini API paid tier for higher limits

### Audio loading failures
- Install FFmpeg: `sudo apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)
- Check that audio files are not corrupted

## Development

### Code Style

The codebase follows professional Python standards:
- Type hints on all functions
- Comprehensive docstrings (Google style)
- Proper logging (no print statements)
- Named constants instead of magic numbers
- Context managers for resource management

### Adding New Features

1. Follow existing code patterns
2. Add type hints and docstrings
3. Use the shared `utils.py` for common functionality
4. Implement proper error handling and logging
5. Update this README with new configuration or usage

## License

This project is for educational and personal use.

## Credits

- **Moonshine**: UsefulSensors speech-to-text model
- **Google Gemini**: AI for story segmentation and embeddings
- **TiDB**: Vector database for semantic search
