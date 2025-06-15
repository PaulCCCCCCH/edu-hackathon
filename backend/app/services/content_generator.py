"""Service for generating video content using Gemini."""

import json
import logging
import os
from datetime import datetime
from typing import Any
from uuid import uuid4

from google import genai
from google.genai import types
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

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the content generator service."""
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            msg = "GEMINI_API_KEY is required"
            raise ValueError(msg)
        self.client = genai.Client(api_key=api_key)
        # Use the latest recommended model from the documentation
        self.model_name = "gemini-2.5-flash-preview-05-20"

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

            # Integrate system instruction into the prompt for compatibility
            full_prompt = f"""You are an expert educational content creator. Generate engaging, accurate, and well-structured educational content. Always respond with valid JSON that matches the requested schema.

{prompt}"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                    response_mime_type="application/json"
                )
            )

            try:
                # Ensure response.text is a string
                response_text = str(response.text) if response.text is not None else "{}"
                content_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {e}")
                logger.warning(f"Raw response: {response.text}")
                # Try to fix common JSON issues
                try:
                    # Remove potential trailing characters and try again
                    cleaned_response = str(response.text).strip() if response.text else "{}"
                    cleaned_response = cleaned_response.removesuffix(",")
                    content_data = json.loads(cleaned_response)
                except json.JSONDecodeError:
                    # Fallback to a default structure with longer transcript
                    logger.warning("Using fallback content structure")
                    # Generate a longer transcript that takes 20-40 seconds to read
                    # Average speaking rate: 150 words per minute = 2.5 words per second
                    # For 30 seconds, we need about 75 words
                    fallback_transcript = f"""Welcome to this educational video about {topic}! [PAUSE]
                    
Today, we're diving deep into the fascinating world of {topic}. This is one of the most important concepts you'll learn, and I'm excited to share it with you.

First, let's understand what {topic} really means. At its core, {topic} is about understanding fundamental principles that shape our understanding of the subject. [PAUSE]

The key things to remember are: First, always approach {topic} with curiosity. Second, practice makes perfect when it comes to mastering these concepts. And third, don't be afraid to make mistakes - they're part of the learning process.

By the end of this video, you'll have a solid foundation in {topic} and be ready to explore even more advanced topics. [PAUSE]

Remember to practice what you've learned today, and if you found this helpful, let me know in the comments! Keep learning and stay curious!"""
                    
                    content_data = {
                        "title": f"Master {topic} in 30 Seconds",
                        "transcript": fallback_transcript,
                        "topics": [topic],
                        "duration_seconds": 30,
                        "key_points": ["Core concepts", "Practical applications", "Learning tips"],
                        "visual_cues": ["Opening hook animation", "Key concept graphics", "Summary points"]
                    }

            # Debug: Log the response structure
            logger.info(f"Gemini response type: {type(content_data)}")

            # Handle both list and dict responses
            if isinstance(content_data, list):
                if content_data and isinstance(content_data[0], dict):
                    content_data = content_data[0]
                else:
                    raise ValueError("Invalid response format: list does not contain dict")
            elif not isinstance(content_data, dict):
                raise ValueError(f"Invalid response format: expected dict, got {type(content_data)}")

            # Create VideoContent object
            video_content = VideoContent(
                title=content_data.get("title", f"Educational video about {topic}"),
                transcript=content_data.get("transcript", f"This is an educational video about {topic}."),
                topics=content_data.get("topics", [topic]),
                difficulty_level=difficulty_level,
                duration_seconds=content_data.get("duration_seconds", 30),
                style=style,
                key_points=content_data.get("key_points", []),
                visual_cues=content_data.get("visual_cues", []),
                metadata={
                    "target_audience": target_audience,
                    "language": "en",
                    "generation_model": self.model_name
                }
            )

            logger.info(f"Generated video transcript for topic: {topic}")
            return video_content

        except Exception as e:
            logger.exception(f"Error generating video transcript: {e!s}")
            raise

    async def generate_video_batch(
        self,
        interests: list[str],
        learning_style: str = "mixed",
        difficulty_preference: str = "intermediate",
        recent_topics: list[str] | None = None,
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

            # Integrate system instruction into the prompt for compatibility
            full_prompt = f"""You are an expert educational content strategist. Create diverse, engaging educational video concepts that match user interests and learning preferences. Always respond with valid JSON array.

{prompt}"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=4096,
                    response_mime_type="application/json"
                )
            )

            response_text = str(response.text) if response.text is not None else "[]"
            concepts_data = json.loads(response_text)
            concepts = [VideoConcept(**concept) for concept in concepts_data]

            logger.info(f"Generated batch of {len(concepts)} video concepts")
            return concepts

        except Exception as e:
            logger.exception(f"Error generating video batch: {e!s}")
            raise

    async def recommend_content(
        self,
        liked_topics: list[str],
        disliked_topics: list[str] | None = None,
        viewing_patterns: dict[str, Any] | None = None,
        learning_goals: list[str] | None = None,
        time_spent: dict[str, float] | None = None,
        available_content: list[dict[str, Any]] | None = None,
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

            # Integrate system instruction into the prompt for compatibility
            full_prompt = f"""You are an AI learning recommendation system. Analyze user learning patterns and preferences to provide personalized content recommendations. Always respond with valid JSON array of recommendations.

{prompt}"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=3072,
                    response_mime_type="application/json"
                )
            )

            response_text = str(response.text) if response.text is not None else "[]"
            recommendations_data = json.loads(response_text)
            recommendations = [ContentRecommendation(**rec) for rec in recommendations_data]

            logger.info(f"Generated {len(recommendations)} content recommendations")
            return recommendations

        except Exception as e:
            logger.exception(f"Error generating content recommendations: {e!s}")
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
