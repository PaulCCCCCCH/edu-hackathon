"""Shared Pydantic schemas used by the EduTok backend."""

from typing import List
from pydantic import BaseModel

class UserBackground(BaseModel):
    job: str
    education_level: str
    interests: List[str]

class TopicsRequest(BaseModel):
    topics: List[str]

class VideoResponse(BaseModel):
    video_url: str
    transcript: str
