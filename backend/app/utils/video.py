"""Utility helpers related to video handling with real audio generation."""

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from ..schemas import VideoResponse
from ..services.audio_service import AudioService
from ..services.content_generator import ContentGeneratorService
from ..services.video_service import VideoService
from . import state

logger = logging.getLogger(__name__)

# Directory: root/storage  (created at runtime if needed)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
STORAGE_DIR = ROOT_DIR / "storage"
VIDEOS_DIR = STORAGE_DIR / "videos"
AUDIO_DIR = STORAGE_DIR / "audio"

# In-memory video generation status tracking
video_generation_status = {}


def _ensure_dirs() -> None:
    """Create storage directory tree if it does not exist."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


async def generate_and_store_videos(count: int = 2, topics=None, base_url: str | None = None) -> list[VideoResponse]:
    """Generate *count* videos with real audio, store them in memory and on disk, and return them."""
    topics = topics or ["your chosen topic"]
    _ensure_dirs()

    # Initialize services
    content_service = ContentGeneratorService()
    audio_service = AudioService(storage_path=str(AUDIO_DIR))
    video_service = VideoService(storage_path=str(VIDEOS_DIR))

    responses: list[VideoResponse] = []

    for i in range(count):
        topic = topics[i % len(topics)]
        video_id = str(uuid4())

        try:
            # Step 1: Generate content/transcript using Gemini
            logger.info(f"Generating content for topic: {topic}")
            video_content = await content_service.generate_video_transcript(
                topic=topic,
                difficulty_level=state.user_background.education_level if state.user_background else "intermediate",
                target_audience=state.user_background.interests[0] if state.user_background and state.user_background.interests else "general learners",
                style="explanation"
            )

            # Step 2: Generate audio from transcript
            logger.info(f"Generating audio for video {video_id}")
            audio_result = await audio_service.generate_audio_from_transcript(
                transcript=video_content.transcript,
                video_id=video_id,
                style=video_content.style
            )

            # Step 3: Generate video (use actual audio duration, not estimated duration)
            actual_duration = audio_result.duration_seconds if audio_result.duration_seconds else video_content.duration_seconds
            logger.info(f"Generating video for video {video_id} with duration {actual_duration}s (audio: {audio_result.duration_seconds}s)")
            video_result = await video_service.generate_video_from_transcript(
                transcript=video_content.transcript,
                video_id=video_id,
                audio_file_path=audio_result.audio_file_path,
                style=video_content.style,
                duration_seconds=actual_duration
            )

            # Create the response with proper URLs
            video_url = f"{base_url}/storage/videos/{Path(video_result.video_file_path).name}" if base_url else f"/storage/videos/{Path(video_result.video_file_path).name}"
            audio_url = f"{base_url}/storage/audio/{Path(audio_result.audio_file_path).name}" if base_url else f"/storage/audio/{Path(audio_result.audio_file_path).name}"

            resp = VideoResponse(
                video_url=video_url,
                transcript=video_content.transcript,
                title=video_content.title,
                audio_url=audio_url,
                duration_seconds=actual_duration,
                topics=video_content.topics,
                metadata={
                    "video_id": video_id,
                    "style": video_content.style,
                    "difficulty_level": video_content.difficulty_level,
                    "voice_used": audio_result.config_used.voice_name,
                    "generation_method": video_result.generation_method
                }
            )

            responses.append(resp)
            state.video_memory.append(resp)

            logger.info(f"Successfully generated video {i+1}/{count}: {video_id}")

        except Exception as e:
            logger.exception(f"Error generating video {i+1}: {e}")
            # Create a minimal response even on error
            error_msg = str(e).replace('"', "'")[:200]  # Sanitize and truncate error message
            resp = VideoResponse(
                video_url=f"{base_url}/static/error.mp4" if base_url else "/static/error.mp4",
                transcript=f"Error generating content for {topic}: {error_msg}"
            )
            responses.append(resp)
            state.video_memory.append(resp)

    return responses


async def start_concurrent_generation(transcript: str, topics: list[str] | None = None) -> str:
    """Start generating video and audio concurrently, return video ID immediately."""
    video_id = str(uuid4())
    topics = topics or ["educational content"]

    # Track generation status
    video_generation_status[video_id] = {
        "status": "generating",
        "transcript": transcript,
        "audio_ready": False,
        "video_ready": False,
        "error": None
    }

    # Start async generation task
    asyncio.create_task(_generate_video_async(video_id, transcript, topics))

    return video_id


async def _generate_video_async(video_id: str, transcript: str, topics: list[str]) -> None:
    """Generate video and audio asynchronously."""
    try:
        _ensure_dirs()

        # Initialize services
        audio_service = AudioService(storage_path=str(AUDIO_DIR))
        video_service = VideoService(storage_path=str(VIDEOS_DIR))

        # Generate audio and video concurrently
        audio_task = asyncio.create_task(
            audio_service.generate_audio_from_transcript(
                transcript=transcript,
                video_id=video_id,
                style="explanation"
            )
        )

        video_task = asyncio.create_task(
            video_service.generate_video_from_transcript(
                transcript=transcript,
                video_id=video_id,
                style="explanation",
                duration_seconds=45.0
            )
        )

        # Wait for both to complete
        audio_result, video_result = await asyncio.gather(audio_task, video_task)

        # Update status
        video_generation_status[video_id].update({
            "status": "completed",
            "audio_ready": True,
            "video_ready": True,
            "audio_path": audio_result.audio_file_path,
            "video_path": video_result.video_file_path,
            "metadata": {
                "voice_used": audio_result.config_used.voice_name,
                "duration": audio_result.duration_seconds,
                "generation_method": video_result.generation_method
            }
        })

        # Create and store VideoResponse
        resp = VideoResponse(
            video_url=f"/storage/videos/{Path(video_result.video_file_path).name}",
            transcript=transcript,
            audio_url=f"/storage/audio/{Path(audio_result.audio_file_path).name}",
            metadata={
                "video_id": video_id,
                "topics": topics,
                "status": "completed"
            }
        )
        state.video_memory.append(resp)

    except Exception as e:
        logger.exception(f"Error in async video generation for {video_id}: {e}")
        video_generation_status[video_id].update({
            "status": "error",
            "error": str(e)
        })


def get_sample_video() -> str:
    """Return a random placeholder video URL from SAMPLE_URLS."""
    return random.choice(SAMPLE_URLS)

def get_video_status(video_id: str) -> dict | None:
    """Get the status of a video generation task."""
    return video_generation_status.get(video_id)
