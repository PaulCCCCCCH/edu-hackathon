"""Service for generating video content using GPT-4o."""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel, Field

from ..prompts.video_generation_prompts import (
    BATCH_GENERATION_PROMPT,
    RECOMMENDATION_PROMPT,
    TRANSCRIPT_GENERATION_PROMPT,
)


logger = logging.getLogger(__name__)


class VideoContent(BaseModel):
    """Generated video content model."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    transcript: str
    topics: list[str]
    difficulty_level: str
    duration_seconds: float
    style: str
    key_points: list[str] = Field(default_factory=list)
    visual_cues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)


class VideoConcept(BaseModel):
    """Video concept for batch generation."""

    topic: str
    difficulty_level: str
    style: str
    target_audience: str
    connection_to_interests: str


class ContentRecommendation(BaseModel):
    """Content recommendation model."""

    content_id: str
    recommendation_score: float
    reasoning: str
    learning_value: str
    difficulty_match: str


class ContentGeneratorService:
    """Service for generating educational video content."""

    def __init__(self, api_key: str | None = None):
        """Initialize the content generator service."""
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = "gpt-4o"

    async def generate_video_transcript(
        self,
        topic: str,
        difficulty_level: str = "intermediate",
        target_audience: str = "general learners",
        style: str = "explanation"
    ) -> VideoContent:
        """Generate a video transcript with metadata."""
        try:
            prompt = TRANSCRIPT_GENERATION_PROMPT.format(
                topic=topic,
                difficulty_level=difficulty_level,
                target_audience=target_audience,
                style=style
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational content creator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )

            content_data = json.loads(response.choices[0].message.content)

            # Create VideoContent object
            video_content = VideoContent(
                title=content_data["title"],
                transcript=content_data["transcript"],
                topics=content_data["topics"],
                difficulty_level=difficulty_level,
                duration_seconds=content_data["duration_seconds"],
                style=style,
                key_points=content_data.get("key_points", []),
                visual_cues=content_data.get("visual_cues", []),
                metadata={
                    "target_audience": target_audience,
                    "language": "en",
                    "generation_model": self.model
                }
            )

            logger.info(f"Generated video transcript for topic: {topic}")
            return video_content

        except Exception as e:
            logger.error(f"Error generating video transcript: {e!s}")
            raise

    async def generate_video_batch(
        self,
        interests: list[str],
        learning_style: str = "mixed",
        difficulty_preference: str = "intermediate",
        recent_topics: list[str] = None,
        batch_size: int = 20
    ) -> list[VideoConcept]:
        """Generate a batch of video concepts."""
        try:
            recent_topics = recent_topics or []

            prompt = BATCH_GENERATION_PROMPT.format(
                batch_size=batch_size,
                interests=", ".join(interests),
                learning_style=learning_style,
                difficulty_preference=difficulty_preference,
                recent_topics=", ".join(recent_topics)
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational content strategist."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8
            )

            concepts_data = json.loads(response.choices[0].message.content)
            concepts = [VideoConcept(**concept) for concept in concepts_data]

            logger.info(f"Generated batch of {len(concepts)} video concepts")
            return concepts

        except Exception as e:
            logger.error(f"Error generating video batch: {e!s}")
            raise

    async def recommend_content(
        self,
        liked_topics: list[str],
        disliked_topics: list[str] = None,
        viewing_patterns: dict[str, Any] = None,
        learning_goals: list[str] = None,
        time_spent: dict[str, float] = None,
        available_content: list[dict[str, Any]] = None,
        num_recommendations: int = 10
    ) -> list[ContentRecommendation]:
        """Generate content recommendations based on user preferences."""
        try:
            disliked_topics = disliked_topics or []
            viewing_patterns = viewing_patterns or {}
            learning_goals = learning_goals or []
            time_spent = time_spent or {}
            available_content = available_content or []

            prompt = RECOMMENDATION_PROMPT.format(
                liked_topics=", ".join(liked_topics),
                disliked_topics=", ".join(disliked_topics),
                viewing_patterns=json.dumps(viewing_patterns),
                learning_goals=", ".join(learning_goals),
                time_spent=json.dumps(time_spent),
                available_content=json.dumps(available_content),
                num_recommendations=num_recommendations
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI learning recommendation system."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            recommendations_data = json.loads(response.choices[0].message.content)
            recommendations = [ContentRecommendation(**rec) for rec in recommendations_data]

            logger.info(f"Generated {len(recommendations)} content recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating content recommendations: {e!s}")
            raise

    async def generate_full_video_content(
        self,
        topic: str,
        difficulty_level: str = "intermediate",
        target_audience: str = "general learners",
        style: str = "explanation"
    ) -> VideoContent:
        """Generate complete video content including transcript and metadata."""
        return await self.generate_video_transcript(
            topic=topic,
            difficulty_level=difficulty_level,
            target_audience=target_audience,
            style=style
        )
