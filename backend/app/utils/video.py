"""Utility helpers related to video handling.

This module is responsible for persisting **fake** video files to disk so that
they can be served by FastAPI's static file handler during demos.
It also creates suitable `VideoResponse` objects and appends them to
`state.video_memory` so that other endpoints can fetch videos by index.
"""

import shutil
import urllib.request
from pathlib import Path
from uuid import uuid4
from typing import List

from ..schemas import VideoResponse
from . import transcript as transcript_utils
from . import state

# Public-domain sample video that will be duplicated to generate additional files.
SAMPLE_URL = "https://www.w3schools.com/html/mov_bbb.mp4"

# Directory: backend/static/videos  (created at runtime if needed)
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = BACKEND_ROOT / "static"
VIDEOS_DIR = STATIC_DIR / "videos"
MASTER_SAMPLE = VIDEOS_DIR / "_sample_master.mp4"


def _ensure_dirs() -> None:
    """Create static/videos directory tree if it does not exist."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_master_sample() -> None:
    """Download the sample video once so that subsequent copies are quick."""
    if not MASTER_SAMPLE.exists():
        _ensure_dirs()
        urllib.request.urlretrieve(SAMPLE_URL, MASTER_SAMPLE)


def save_fake_videos(count: int = 2, base_url: str | None = None) -> List[str]:
    """Save *count* fake videos under *static/videos* and return their URLs."""
    _ensure_master_sample()
    urls: List[str] = []
    for _ in range(count):
        filename = f"{uuid4().hex}.mp4"
        dest_path = VIDEOS_DIR / filename
        shutil.copyfile(MASTER_SAMPLE, dest_path)
        # URL that the frontend can fetch via FastAPI static handler
        path_part = f"/static/videos/{filename}"
        if base_url:
            urls.append(f"{base_url}{path_part}")
        else:
            urls.append(path_part)
    return urls


def generate_and_store_videos(count: int = 2, topics=None, base_url: str | None = None) -> List[VideoResponse]:
    """Generate *count* videos, store them in memory and on disk, and return them."""
    topics = topics or ["your chosen topic"]
    urls = save_fake_videos(count, base_url=base_url)

    responses: List[VideoResponse] = []
    for i, url in enumerate(urls):
        topic = topics[i % len(topics)]
        transcript = (
            transcript_utils.generate_transcript(state.user_background, topic)
            if state.user_background
            else f"Brief overview on {topic}."
        )
        resp = VideoResponse(video_url=url, transcript=transcript)
        responses.append(resp)
        state.video_memory.append(resp)
    return responses


def get_sample_video() -> str:
    """Return a placeholder video URL (public-domain sample)."""
    return SAMPLE_URL
