"""Router for video streaming and progressive loading."""

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .utils import video as video_utils


router = APIRouter()

@router.get("/stream_video/{video_id}")
async def stream_video(video_id: str):
    """Stream video as soon as audio is ready, with progressive loading."""
    status_data = video_utils.get_video_status(video_id)

    if not status_data:
        raise HTTPException(status_code=404, detail="Video not found")

    async def generate_stream():
        """Generate video stream with immediate audio playback."""
        # Return transcript immediately
        yield f"data: {{'type': 'transcript', 'content': '{status_data['transcript']}'}}\n\n"

        # Wait for audio to be ready
        while not status_data.get("audio_ready", False):
            await asyncio.sleep(0.1)
            status_data = video_utils.get_video_status(video_id)
            if not status_data:
                return

        # Stream audio as soon as it's ready
        yield f"data: {{'type': 'audio', 'url': '{status_data['audio_url']}'}}\n\n"

        # Continue streaming video updates
        while not status_data.get("video_ready", False):
            await asyncio.sleep(0.5)
            status_data = video_utils.get_video_status(video_id)
            if not status_data:
                return
            yield "data: {'type': 'progress', 'status': 'generating_video'}\n\n"

        # Stream video when ready
        yield f"data: {{'type': 'video', 'url': '{status_data['video_url']}'}}\n\n"
        yield "data: {'type': 'complete'}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@router.get("/quick_start/{video_id}")
async def quick_start_video(video_id: str):
    """Get immediate playback data - transcript first, then audio when ready."""
    status_data = video_utils.get_video_status(video_id)

    if not status_data:
        raise HTTPException(status_code=404, detail="Video not found")

    response_data = {
        "video_id": video_id,
        "transcript": status_data["transcript"],
        "can_start_immediately": True
    }

    # Add audio if ready
    if status_data.get("audio_ready"):
        response_data["audio_url"] = status_data["audio_url"]
        response_data["audio_ready"] = True

    # Add video if ready
    if status_data.get("video_ready"):
        response_data["video_url"] = status_data["video_url"]
        response_data["video_ready"] = True

    return response_data
