"""Simple in-memory state shared across routers."""

from typing import Optional, List

from ..schemas import UserBackground, VideoResponse

# This will reset whenever the server restarts
user_background: Optional[UserBackground] = None

# Holds generated videos (saved as VideoResponse objects) for the current runtime
video_memory: List[VideoResponse] = []
