"""Simple in-memory state shared across routers."""


from ..schemas import UserBackground, VideoResponse


# This will reset whenever the server restarts
user_background: UserBackground | None = None

# Holds generated videos (saved as VideoResponse objects) for the current runtime
video_memory: list[VideoResponse] = []
