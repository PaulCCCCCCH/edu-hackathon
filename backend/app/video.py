"""Router handling video generation."""

from fastapi import APIRouter, HTTPException

from typing import List

from .schemas import TopicsRequest, VideoResponse
from .utils import state, transcript as transcript_utils, video as video_utils


router = APIRouter()

from fastapi import Request

@router.post("/generate_videos", response_model=List[VideoResponse])
async def generate_videos(req: TopicsRequest, request: Request):
    """Generate *two* videos and transcripts; persist them & return the list."""
    if state.user_background is None:
        raise HTTPException(status_code=400, detail="User background not submitted yet.")

    # Generate two videos and store
    base_url = str(request.base_url).rstrip("/")
    return video_utils.generate_and_store_videos(count=2, topics=req.topics, base_url=base_url)


@router.get("/fetch_video/{idx}", response_model=VideoResponse)
async def fetch_video(idx: int, request: Request):
    """Return the video at *idx*, generating more if needed."""
    if idx < 0:
        raise HTTPException(status_code=400, detail="Index must be non-negative.")

    # If we don't have enough videos, generate more in batches of 2.
    while idx >= len(state.video_memory):
        # fallback to generic topic list when none available
        base_url = str(request.base_url).rstrip("/")
        video_utils.generate_and_store_videos(count=2, base_url=base_url)

    return state.video_memory[idx]
