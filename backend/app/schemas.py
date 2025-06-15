"""Shared Pydantic schemas used by the EduTok backend."""


from pydantic import BaseModel


class UserBackground(BaseModel):
    job: str
    education_level: str
    interests: list[str]

class TopicsRequest(BaseModel):
    topics: list[str]

class VideoResponse(BaseModel):
    video_url: str
    transcript: str
