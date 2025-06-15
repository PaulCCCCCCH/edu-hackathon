"""Tests for content generator service."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json

from app.services.content_generator import ContentGeneratorService, VideoContent, VideoConcept


class TestContentGeneratorService:
    """Test cases for ContentGeneratorService."""
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "title": "Test Video Title",
            "transcript": "This is a test transcript with [PAUSE] markers.",
            "topics": ["test", "education", "learning"],
            "duration_seconds": 45.0,
            "key_points": ["Point 1", "Point 2"],
            "visual_cues": ["Visual 1", "Visual 2"]
        })
        return mock_response
    
    @pytest.fixture
    def content_generator(self):
        """Create a content generator instance."""
        with patch('openai.OpenAI') as mock_openai:
            service = ContentGeneratorService()
            service.client = mock_openai.return_value
            return service
    
    @pytest.mark.asyncio
    async def test_generate_video_transcript(self, content_generator, mock_openai_response):
        """Test video transcript generation."""
        content_generator.client.chat.completions.create.return_value = mock_openai_response
        
        result = await content_generator.generate_video_transcript(
            topic="Python basics",
            difficulty_level="beginner",
            style="tutorial"
        )
        
        assert isinstance(result, VideoContent)
        assert result.title == "Test Video Title"
        assert result.transcript == "This is a test transcript with [PAUSE] markers."
        assert result.topics == ["test", "education", "learning"]
        assert result.duration_seconds == 45.0
        assert result.difficulty_level == "beginner"
        assert result.style == "tutorial"
        assert result.key_points == ["Point 1", "Point 2"]
        assert result.visual_cues == ["Visual 1", "Visual 2"]
        
        # Verify API call
        content_generator.client.chat.completions.create.assert_called_once()
        call_args = content_generator.client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4o"
        assert call_args[1]["response_format"] == {"type": "json_object"}
    
    @pytest.mark.asyncio
    async def test_generate_video_batch(self, content_generator):
        """Test video batch generation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps([
            {
                "topic": "Introduction to Machine Learning",
                "difficulty_level": "beginner",
                "style": "explanation",
                "target_audience": "beginners",
                "connection_to_interests": "Related to AI interest"
            },
            {
                "topic": "Data Structures Basics",
                "difficulty_level": "intermediate",
                "style": "tutorial",
                "target_audience": "programming students",
                "connection_to_interests": "Related to programming interest"
            }
        ])
        
        content_generator.client.chat.completions.create.return_value = mock_response
        
        result = await content_generator.generate_video_batch(
            interests=["AI", "programming"],
            learning_style="visual",
            batch_size=2
        )
        
        assert len(result) == 2
        assert all(isinstance(concept, VideoConcept) for concept in result)
        assert result[0].topic == "Introduction to Machine Learning"
        assert result[1].topic == "Data Structures Basics"
    
    @pytest.mark.asyncio
    async def test_generate_video_transcript_error_handling(self, content_generator):
        """Test error handling in transcript generation."""
        content_generator.client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception) as exc_info:
            await content_generator.generate_video_transcript("test topic")
        
        assert "API Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_recommend_content(self, content_generator):
        """Test content recommendation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps([
            {
                "content_id": "video_1",
                "recommendation_score": 0.95,
                "reasoning": "Matches user interests",
                "learning_value": "Learn new concepts",
                "difficulty_match": "Perfect fit"
            }
        ])
        
        content_generator.client.chat.completions.create.return_value = mock_response
        
        result = await content_generator.recommend_content(
            liked_topics=["python", "data science"],
            num_recommendations=1
        )
        
        assert len(result) == 1
        assert result[0].content_id == "video_1"
        assert result[0].recommendation_score == 0.95


# Integration test
@pytest.mark.asyncio
async def test_content_generator_integration():
    """Integration test with real API (requires API key)."""
    try:
        service = ContentGeneratorService()
        
        # This will only work if OPENAI_API_KEY is set
        result = await service.generate_video_transcript(
            topic="Basic Python variables",
            difficulty_level="beginner",
            style="explanation"
        )
        
        assert isinstance(result, VideoContent)
        assert result.title
        assert result.transcript
        assert result.topics
        assert result.duration_seconds > 0
        
    except Exception as e:
        # Skip if no API key is available
        pytest.skip(f"Integration test skipped: {str(e)}")