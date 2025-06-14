"""Router handling video generation."""

from fastapi import APIRouter, HTTPException

from .schemas import TopicsRequest, VideoResponse
from .utils import state
from .utils import transcript as transcript_utils
from .utils import video as video_utils

router = APIRouter()

@router.post("/generate_video", response_model=VideoResponse)
async def generate_video(req: TopicsRequest):
    """Generate a video (placeholder) and transcript based on user background."""
    if state.user_background is None:
        raise HTTPException(status_code=400, detail="User background not submitted yet.")

    primary_topic = req.topics[0] if req.topics else "your chosen topic"
    transcript = transcript_utils.generate_transcript(state.user_background, primary_topic)
    sample_video = video_utils.get_sample_video()

    return VideoResponse(video_url=sample_video, transcript=transcript)
