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

import random

# Public-domain sample videos that will be duplicated to generate additional files.
SAMPLE_URLS = [
    "https://www.w3schools.com/html/mov_bbb.mp4",  # Big Buck Bunny (W3Schools)
    "https://www.w3schools.com/html/movie.mp4",     # Bear (W3Schools)
    "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_2mb.mp4",
    "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_5mb.mp4",
]

# Directory: backend/static/videos  (created at runtime if needed)
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = BACKEND_ROOT / "static"
VIDEOS_DIR = STATIC_DIR / "videos"
MASTER_SAMPLES = [VIDEOS_DIR / f"_sample_master_{i}.mp4" for i in range(len(SAMPLE_URLS))]


def _ensure_dirs() -> None:
    """Create static/videos directory tree if it does not exist."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_master_samples() -> None:
    """Download each sample video once so that subsequent copies are quick."""
    _ensure_dirs()
    for url, path in zip(SAMPLE_URLS, MASTER_SAMPLES):
        if not path.exists():
            urllib.request.urlretrieve(url, path)


def save_fake_videos(count: int = 2, base_url: str | None = None) -> List[str]:
    """Save *count* fake videos under *static/videos* and return their URLs."""
    _ensure_master_samples()
    urls: List[str] = []
    for _ in range(count):
        filename = f"{uuid4().hex}.mp4"
        dest_path = VIDEOS_DIR / filename
        # choose random master sample file to duplicate
        master_path = random.choice(MASTER_SAMPLES)
        shutil.copyfile(master_path, dest_path)
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
    """Return a random placeholder video URL from SAMPLE_URLS."""
    return random.choice(SAMPLE_URLS)
