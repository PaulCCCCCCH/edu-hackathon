"""Router handling video generation."""

import os

from fastapi import APIRouter, HTTPException

from .schemas import TopicsRequest, VideoResponse
from .utils import state, video as video_utils


router = APIRouter()

from fastapi import Request


@router.post("/generate_videos", response_model=list[VideoResponse])
async def generate_videos(req: TopicsRequest, request: Request):
    """Generate videos based on VIDEOS_PER_BATCH env var; persist them & return the list."""
    if state.user_background is None:
        raise HTTPException(status_code=400, detail="User background not submitted yet.")

    # Get batch size from environment variable
    videos_per_batch = int(os.getenv("VIDEOS_PER_BATCH", "1"))
    base_url = str(request.base_url).rstrip("/")
    return await video_utils.generate_and_store_videos(count=videos_per_batch, topics=req.topics, base_url=base_url)


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


@router.get("/video_stats")
async def get_video_stats():
    """Get statistics about video generation."""
    from pathlib import Path

    storage_path = Path("storage")

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
