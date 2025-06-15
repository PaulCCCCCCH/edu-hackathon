"""Shared Pydantic schemas used by the EduTok backend."""

from datetime import datetime

from pydantic import BaseModel


class UserBackground(BaseModel):
    description: str


class UserPreferences(BaseModel):
    preferred_video_length: int | None = 60
    preferred_difficulty_levels: list[str] = ["beginner", "intermediate"]
    preferred_styles: list[str] = ["explanation", "tutorial"]
    topics_to_avoid: list[str] = []


class WatchedVideo(BaseModel):
    video_id: str
    watched_duration: int
    completion_rate: float
    watched_at: datetime


class TopicEngagement(BaseModel):
    total_videos: int = 0
    average_completion_rate: float = 0.0
    likes: int = 0
    last_engaged: datetime | None = None


class InteractionHistory(BaseModel):
    liked_videos: list[str] = []
    watched_videos: list[WatchedVideo] = []
    topic_engagement: dict[str, TopicEngagement] = {}


class UserProfile(BaseModel):
    user_id: str
    profile: UserBackground
    preferences: UserPreferences = UserPreferences()
    interaction_history: InteractionHistory = InteractionHistory()
    created_at: datetime
    updated_at: datetime | None = None


class TopicsRequest(BaseModel):
    topics: list[str]


class VideoResponse(BaseModel):
    video_url: str
    transcript: str
    title: str | None = None
    audio_url: str | None = None
    duration_seconds: float | None = None
    topics: list[str] | None = None
    metadata: dict | None = None


class VideoMetadata(BaseModel):
    generated_at: datetime
    target_audience: str
    language: str = "en"
    generation_prompt: str | None = None
    processing_status: str = "pending"
    file_paths: dict[str, str] | None = None


class BatchVideo(BaseModel):
    id: str
    title: str
    transcript: str
    topics: list[str]
    difficulty_level: str
    duration_seconds: int
    style: str
    metadata: VideoMetadata


class GenerationContext(BaseModel):
    user_background: UserBackground
    requested_topics: list[str]
    recommended_topics: list[str] = []


class GenerationSettings(BaseModel):
    model_version: str = "gemini-1.5-flash"
    batch_size: int = 20
    diversity_factor: float = 0.7


class BatchMetadata(BaseModel):
    created_at: datetime
    generation_completed_at: datetime | None = None
    total_videos: int
    completed_videos: int = 0
    failed_videos: int = 0
    batch_status: str = "pending"
    generation_settings: GenerationSettings


class VideoBatch(BaseModel):
    batch_id: str
    user_id: str
    generation_context: GenerationContext
    videos: list[BatchVideo]
    batch_metadata: BatchMetadata
