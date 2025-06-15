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
        video_id = str(uuid4())

        try:
            # Step 1: Select concept based on user background
            logger.info("Selecting concept based on user background")
            concept_selection = await content_service.select_concept_for_user(
                user_description=state.user_background.description if state.user_background else "General learner interested in educational content"
            )
            
            # Step 2: Generate content/transcript using the selected concept
            logger.info(f"Generating content for concept: {concept_selection.selected_concept}")
            video_content = await content_service.generate_video_transcript(
                topic=concept_selection.selected_concept,
                difficulty_level=concept_selection.difficulty_level,
                target_audience=concept_selection.target_audience,
                style="explanation"
            )

            # Step 3: Generate audio from transcript
            logger.info(f"Generating audio for video {video_id}")
            audio_result = await audio_service.generate_audio_from_transcript(
                transcript=video_content.transcript,
                video_id=video_id,
                style=video_content.style
            )

            # Step 4: Generate video using actual audio duration as target
            # The final video MUST match the full audio duration
            audio_duration = audio_result.duration_seconds if audio_result.duration_seconds else video_content.duration_seconds
            logger.info(f"Generating video for {video_id} to match audio duration: {audio_duration:.2f}s")
            video_result = await video_service.generate_video_from_transcript(
                transcript=video_content.transcript,
                video_id=video_id,
                audio_file_path=audio_result.audio_file_path,
                style=video_content.style,
                duration_seconds=audio_duration  # This ensures the final video will be the same length as the audio
            )

            # Create the response with proper URLs
            video_url = f"{base_url}/storage/videos/{Path(video_result.video_file_path).name}" if base_url else f"/storage/videos/{Path(video_result.video_file_path).name}"
            audio_url = f"{base_url}/storage/audio/{Path(audio_result.audio_file_path).name}" if base_url else f"/storage/audio/{Path(audio_result.audio_file_path).name}"

            resp = VideoResponse(
                video_url=video_url,
                transcript=video_content.transcript,
                title=video_content.title,
                audio_url=audio_url,
                duration_seconds=audio_duration,  # Use the audio duration as the final video duration
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
                transcript=f"Error generating content: {error_msg}"
            )
            responses.append(resp)
            state.video_memory.append(resp)

    return responses


async def start_streaming_generation(count: int = 3, topics=None, base_url: str | None = None) -> list[str]:
    """Start generating videos one by one, return video IDs immediately."""
    topics = topics or ["mathematics", "science", "history"]
    video_ids = []
    
    for i in range(count):
        topic = topics[i % len(topics)]
        video_id = str(uuid4())
        video_ids.append(video_id)
        
        # Track generation status
        video_generation_status[video_id] = {
            "status": "generating",
            "topic": topic,
            "audio_ready": False,
            "video_ready": False,
            "error": None,
            "position": i
        }
        
        # Start async generation with delay to space them out
        asyncio.create_task(_generate_single_video_streaming(video_id, topic, base_url, delay=i * 5))
    
    return video_ids


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
    import random
    SAMPLE_URLS = ["/static/sample1.mp4", "/static/sample2.mp4"]
    return random.choice(SAMPLE_URLS)

async def _generate_single_video_streaming(video_id: str, topic: str, base_url: str | None = None, delay: int = 0) -> None:
    """Generate a single video asynchronously for streaming."""
    if delay > 0:
        await asyncio.sleep(delay)
    
    try:
        _ensure_dirs()
        
        # Initialize services
        content_service = ContentGeneratorService()
        audio_service = AudioService(storage_path=str(AUDIO_DIR))
        video_service = VideoService(storage_path=str(VIDEOS_DIR))
        
        logger.info(f"Starting generation for video {video_id}, topic: {topic}")
        
        # Step 1: Select concept based on user background
        logger.info("Selecting concept based on user background")
        concept_selection = await content_service.select_concept_for_user(
            user_description=state.user_background.description if state.user_background else "General learner interested in educational content"
        )
        
        # Step 2: Generate content/transcript using the selected concept
        video_content = await content_service.generate_video_transcript(
            topic=concept_selection.selected_concept,
            difficulty_level=concept_selection.difficulty_level,
            target_audience=concept_selection.target_audience,
            style="explanation"
        )
        
        # Update status
        video_generation_status[video_id]["transcript"] = video_content.transcript
        video_generation_status[video_id]["title"] = video_content.title
        
        # Step 3: Generate audio
        logger.info(f"Generating audio for video {video_id}")
        audio_result = await audio_service.generate_audio_from_transcript(
            transcript=video_content.transcript,
            video_id=video_id,
            style=video_content.style
        )
        
        video_generation_status[video_id]["audio_ready"] = True
        
        # Step 4: Generate video
        audio_duration = audio_result.duration_seconds if audio_result.duration_seconds else video_content.duration_seconds
        logger.info(f"Generating video for {video_id}")
        video_result = await video_service.generate_video_from_transcript(
            transcript=video_content.transcript,
            video_id=video_id,
            audio_file_path=audio_result.audio_file_path,
            style=video_content.style,
            duration_seconds=audio_duration
        )
        
        # Create the response
        video_url = f"{base_url}/storage/videos/{Path(video_result.video_file_path).name}" if base_url else f"/storage/videos/{Path(video_result.video_file_path).name}"
        audio_url = f"{base_url}/storage/audio/{Path(audio_result.audio_file_path).name}" if base_url else f"/storage/audio/{Path(audio_result.audio_file_path).name}"
        
        resp = VideoResponse(
            video_url=video_url,
            transcript=video_content.transcript,
            title=video_content.title,
            audio_url=audio_url,
            duration_seconds=audio_duration,
            topics=video_content.topics,
            metadata={
                "video_id": video_id,
                "style": video_content.style,
                "difficulty_level": video_content.difficulty_level,
                "voice_used": audio_result.config_used.voice_name,
                "generation_method": video_result.generation_method
            }
        )
        
        # Update status and add to memory
        video_generation_status[video_id].update({
            "status": "completed",
            "video_ready": True,
            "video_response": resp
        })
        
        state.video_memory.append(resp)
        logger.info(f"Successfully generated video {video_id}")
        
    except Exception as e:
        logger.exception(f"Error generating video {video_id}: {e}")
        video_generation_status[video_id].update({
            "status": "error",
            "error": str(e)
        })


def get_video_status(video_id: str) -> dict | None:
    """Get the status of a video generation task."""
    return video_generation_status.get(video_id)
