"""Router handling video generation."""

import os
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .schemas import TopicsRequest, VideoResponse
from .utils import state, video as video_utils


router = APIRouter()

from fastapi import Request


class HistoryVideo(BaseModel):
    """Model for a video in history."""
    filename: str
    video_url: str
    audio_url: str | None = None
    topic: str | None = None
    transcript: str | None = None
    created_at: str


@router.post("/generate_videos", response_model=list[VideoResponse])
async def generate_videos(req: TopicsRequest, request: Request):
    """Generate videos based on VIDEOS_PER_BATCH env var; persist them & return the list."""
    if state.user_background is None:
        raise HTTPException(status_code=400, detail="User background not submitted yet.")

    # Get batch size from environment variable
    videos_per_batch = int(os.getenv("VIDEOS_PER_BATCH", "1"))
    base_url = str(request.base_url).rstrip("/")
    return await video_utils.generate_and_store_videos(count=videos_per_batch, topics=req.topics, base_url=base_url)


@router.post("/start_video_generation")
async def start_video_generation(req: TopicsRequest, request: Request):
    """Start generating videos one by one, return immediately with video IDs."""
    if state.user_background is None:
        raise HTTPException(status_code=400, detail="User background not submitted yet.")
    
    videos_per_batch = int(os.getenv("VIDEOS_PER_BATCH", "3"))
    base_url = str(request.base_url).rstrip("/")
    
    # Start generating videos asynchronously, one at a time
    video_ids = await video_utils.start_streaming_generation(
        count=videos_per_batch, 
        topics=req.topics, 
        base_url=base_url
    )
    
    return {"video_ids": video_ids, "status": "generation_started"}


@router.get("/fetch_video/{idx}", response_model=VideoResponse)
async def fetch_video(idx: int, request: Request):
    """Return the video at *idx*, generating more if needed."""
    if idx < 0:
        raise HTTPException(status_code=400, detail="Index must be non-negative.")

    # If we don't have enough videos, generate more based on VIDEOS_PER_BATCH.
    while idx >= len(state.video_memory):
        # fallback to generic topic list when none available
        videos_per_batch = int(os.getenv("VIDEOS_PER_BATCH", "1"))
        base_url = str(request.base_url).rstrip("/")
        await video_utils.generate_and_store_videos(count=videos_per_batch, base_url=base_url)

    return state.video_memory[idx]


@router.get("/video_status/{video_id}")
async def get_video_status(video_id: str):
    """Get the status of a video generation operation."""
    status = video_utils.get_video_status(video_id)
    if not status:
        raise HTTPException(status_code=404, detail="Video ID not found")
    return status


@router.get("/stream_status/{video_id}")
async def get_stream_status(video_id: str):
    """Get the streaming status of a video - returns the video if ready."""
    status = video_utils.get_video_status(video_id)
    if not status:
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    if status.get("status") == "completed" and status.get("video_response"):
        return {
            "status": "ready",
            "video": status["video_response"]
        }
    elif status.get("status") == "error":
        return {
            "status": "error", 
            "error": status.get("error", "Unknown error")
        }
    else:
        return {
            "status": "generating",
            "topic": status.get("topic", "Unknown"),
            "audio_ready": status.get("audio_ready", False),
            "video_ready": status.get("video_ready", False)
        }


@router.get("/video_stats")
async def get_video_stats():
    """Get statistics about video generation."""
    from pathlib import Path

    storage_path = Path("../storage")

    stats = {
        "total_videos_generated": len(state.video_memory),
        "video_files_on_disk": 0,
        "audio_files_on_disk": 0,
        "real_videos": 0,
        "mock_videos": 0
    }

    # Count video files
    video_path = storage_path / "videos"
    if video_path.exists():
        video_files = list(video_path.glob("*.mp4"))
        stats["video_files_on_disk"] = len(video_files)

        # Differentiate real vs mock videos by file size
        for f in video_files:
            size_mb = f.stat().st_size / 1024 / 1024
            if size_mb > 0.1:  # > 100KB likely real video
                stats["real_videos"] += 1
            else:
                stats["mock_videos"] += 1

    # Count audio files
    audio_path = storage_path / "audio"
    if audio_path.exists():
        stats["audio_files_on_disk"] = len(list(audio_path.glob("*.wav")))

    return stats


@router.get("/history", response_model=List[HistoryVideo])
async def get_video_history(request: Request):
    """Get all videos from storage/videos directory."""
    try:
        # Get the storage directory
        base_dir = Path(__file__).resolve().parent.parent.parent
        videos_dir = base_dir / "storage" / "videos"
        audio_dir = base_dir / "storage" / "audio"
        
        if not videos_dir.exists():
            return []
        
        history_videos = []
        base_url = str(request.base_url).rstrip("/")
        
        # Get all video files
        for video_file in videos_dir.glob("*.mp4"):
            try:
                # Get file stats
                stat = video_file.stat()
                created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
                
                # Construct URLs
                video_url = f"{base_url}/storage/videos/{video_file.name}"
                
                # Look for corresponding audio file
                audio_url = None
                for audio_ext in ['.wav', '.mp3']:
                    audio_file = audio_dir / f"{video_file.stem}{audio_ext}"
                    if audio_file.exists():
                        audio_url = f"{base_url}/storage/audio/{audio_file.name}"
                        break
                
                # Extract topic from filename if possible
                topic = video_file.stem.replace("_", " ").replace("-", " ").title()
                if topic.startswith("Video "):
                    topic = topic[6:]  # Remove "Video " prefix
                
                history_video = HistoryVideo(
                    filename=video_file.name,
                    video_url=video_url,
                    audio_url=audio_url,
                    topic=topic,
                    transcript=f"Educational content: {topic}",
                    created_at=created_at
                )
                
                history_videos.append(history_video)
                
            except Exception as e:
                print(f"Error processing video file {video_file}: {e}")
                continue
        
        # Sort by creation time (newest first)
        history_videos.sort(key=lambda x: x.created_at, reverse=True)
        
        return history_videos
        
    except Exception as e:
        print(f"Error getting video history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get video history: {str(e)}")


@router.get("/history/count")
async def get_history_count():
    """Get count of videos in history."""
    try:
        base_dir = Path(__file__).resolve().parent.parent.parent
        videos_dir = base_dir / "storage" / "videos"
        
        if not videos_dir.exists():
            return {"count": 0}
        
        count = len(list(videos_dir.glob("*.mp4")))
        return {"count": count}
        
    except Exception as e:
        print(f"Error getting history count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history count: {str(e)}")
