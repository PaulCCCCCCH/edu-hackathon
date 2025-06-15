# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EduTok Demo** - A TikTok-like educational video platform that generates short educational videos on-demand using AI. This is a full-stack prototype with a FastAPI backend and single-page HTML frontend.

## Development Commands

### Running the Application

```bash
# start the contianer
source edu_hackathon/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --port 8000 --host 0.0.0.0

# Frontend: Open frontend/index.html in browser
```

### Code Quality

```bash
# Lint and format code
ruff check .
ruff format .
```

## Architecture Overview

### Backend Structure (`backend/app/`)

- **FastAPI Application**: Two main routers - `background.py` (user profiles) and `video.py` (video generation)
- **Services Layer**: AI-powered content generation using Google Gemini, TTS, and video APIs
  - `content_generator.py` - Gemini integration for transcript generation
  - `audio_service.py` - Google TTS integration
  - `video_service.py` - Video generation pipeline
- **Utils Layer**: Processing utilities for audio/video manipulation
- **Storage**: File-based storage in `storage/` directory (no database)
- **State Management**: In-memory state via `utils/state.py`
- use type hints 

### Key Technologies

- **Google Generative AI (Gemini)** - Content generation, audio and video
- **Pydantic** - Data validation and serialization
- **FastAPI** - Web framework with automatic OpenAPI docs

### Data Flow

1. User submits background info via `/background` endpoint
2. Frontend requests video via `/generate_video` with topics
3. Backend generates transcript using Gemini + user background
4. Audio/video processing pipeline creates final content
5. Files stored locally in `storage/` directories

### Storage Organization

- `storage/videos/` - Generated video files
- `storage/audio/` - Generated audio files
- `storage/temp/` - Temporary processing files
- `data/` - User data and video batches (JSON)

## Important Notes

- **No Database**: All data stored in-memory or as local files
- **AI-Heavy**: Most functionality depends on Google AI services
- **Hackathon Prototype**: Simplified architecture, production-ready code quality
- **CORS Enabled**: Backend allows all origins for demo purposes
- **Python 3.12**: Target runtime version with comprehensive linting via Ruff
