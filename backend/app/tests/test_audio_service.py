"""Tests for audio service."""

import sys
import os
import tempfile
import shutil
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.services.audio_service import AudioService, AudioConfig, AudioGenerationResult
from app.utils.audio_processing import AudioProcessor


class TestAudioService:
    """Test cases for AudioService."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def audio_service(self, temp_storage):
        """Create an audio service instance."""
        return AudioService(storage_path=temp_storage)
    
    @pytest.mark.asyncio
    async def test_mock_audio_generation(self, audio_service):
        """Test mock audio generation (when Google TTS is not available)."""
        result = await audio_service.generate_audio(
            text="Hello, this is a test transcript for audio generation.",
            output_filename="test_audio.mp3"
        )
        
        assert isinstance(result, AudioGenerationResult)
        assert result.audio_file_path.endswith("test_audio.mp3")
        assert result.duration_seconds > 0
        assert result.file_size_bytes > 0
        assert result.metadata["generation_method"] == "mock"
        
        # Verify file was created
        assert os.path.exists(result.audio_file_path)
    
    @pytest.mark.asyncio
    async def test_generate_audio_from_transcript(self, audio_service):
        """Test generating audio from transcript with style configuration."""
        transcript = "Welcome to this tutorial. [PAUSE] Let's learn about Python variables."
        
        result = await audio_service.generate_audio_from_transcript(
            transcript=transcript,
            video_id="test_video_123",
            style="tutorial"
        )
        
        assert isinstance(result, AudioGenerationResult)
        assert "test_video_123_audio.mp3" in result.audio_file_path
        assert result.config_used.speaking_rate == 0.9  # Tutorial style
        assert result.config_used.voice_name == "en-US-Wavenet-A"
        
        # Verify file was created
        assert os.path.exists(result.audio_file_path)
    
    def test_list_available_voices(self, audio_service):
        """Test listing available voices."""
        voices = audio_service.list_available_voices()
        
        assert isinstance(voices, list)
        assert len(voices) > 0
        
        # Check that mock voices are returned
        voice_names = [voice["name"] for voice in voices]
        assert "en-US-Wavenet-A" in voice_names
        assert "en-US-Wavenet-D" in voice_names
    
    def test_get_audio_info_nonexistent(self, audio_service):
        """Test getting info for non-existent audio file."""
        with pytest.raises(FileNotFoundError):
            audio_service.get_audio_info("nonexistent_file.mp3")
    
    @pytest.mark.asyncio
    async def test_get_audio_info_existing(self, audio_service):
        """Test getting info for existing audio file."""
        # First create an audio file
        result = await audio_service.generate_audio(
            text="Test audio",
            output_filename="info_test.mp3"
        )
        
        # Now get info about it
        info = audio_service.get_audio_info(result.audio_file_path)
        
        assert info["exists"] is True
        assert info["file_size_bytes"] > 0
        assert "created_at" in info


class TestAudioProcessor:
    """Test cases for AudioProcessor."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def audio_processor(self, temp_storage):
        """Create an audio processor instance."""
        return AudioProcessor(storage_path=temp_storage)
    
    def test_clean_transcript_for_tts(self, audio_processor):
        """Test transcript cleaning for TTS."""
        transcript = "Hello there. [PAUSE] This is a test [PAUSE] with multiple pauses"
        cleaned = audio_processor.clean_transcript_for_tts(transcript)
        
        assert "[PAUSE]" not in cleaned
        assert ", " in cleaned  # Pauses should be replaced with commas
        assert cleaned.endswith(".")  # Should end with proper punctuation
    
    def test_get_audio_duration_estimate(self, audio_processor):
        """Test audio duration estimation."""
        text = "This is a test sentence with exactly ten words total."
        duration = audio_processor.get_audio_duration_estimate(text)
        
        assert duration > 0
        assert isinstance(duration, float)
        
        # Should be roughly 4 seconds for 10 words at normal speed
        assert 3 < duration < 8
    
    def test_split_long_text(self, audio_processor):
        """Test splitting long text into chunks."""
        # Create a long text
        long_text = ". ".join([f"This is sentence number {i}" for i in range(100)])
        
        chunks = audio_processor.split_long_text(long_text, max_length=200)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 220 for chunk in chunks)  # Allow some buffer
        assert all(chunk.strip() for chunk in chunks)  # No empty chunks
    
    def test_generate_audio_metadata(self, audio_processor, temp_storage):
        """Test generating audio metadata."""
        # Create a dummy audio file
        audio_file = Path(temp_storage) / "test_audio.mp3"
        audio_file.write_bytes(b"dummy audio content")
        
        text = "This is test audio content"
        metadata = audio_processor.generate_audio_metadata(
            text=text,
            audio_file_path=str(audio_file)
        )
        
        assert metadata["text_metadata"]["word_count"] == 5
        assert metadata["text_metadata"]["character_count"] == len(text)
        assert metadata["file_size_bytes"] > 0
        assert "estimated_duration_seconds" in metadata["text_metadata"]
    
    def test_batch_process_transcripts(self, audio_processor):
        """Test batch processing of transcripts."""
        transcripts = [
            {"id": "video_1", "transcript": "First transcript [PAUSE] with pause"},
            {"id": "video_2", "transcript": "Second transcript text"},
            "Third transcript as plain text"
        ]
        
        processed = audio_processor.batch_process_transcripts(transcripts)
        
        assert len(processed) == 3
        
        # Check first transcript
        assert processed[0]["video_id"] == "video_1"
        assert "[PAUSE]" not in processed[0]["cleaned_text"]
        assert processed[0]["chunk_count"] >= 1
        
        # Check third transcript (plain text)
        assert processed[2]["video_id"] == "video_2"
        assert processed[2]["original_text"] == "Third transcript as plain text"


# Integration test
@pytest.mark.asyncio
async def test_audio_service_integration():
    """Integration test for the complete audio generation workflow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create services
        audio_service = AudioService(storage_path=temp_dir)
        audio_processor = AudioProcessor(storage_path=temp_dir)
        
        # Test data
        transcript = "Welcome to this educational video. [PAUSE] Today we'll learn about Python. [PAUSE] Let's get started!"
        video_id = "integration_test_video"
        
        # Step 1: Process transcript
        cleaned_transcript = audio_processor.clean_transcript_for_tts(transcript)
        assert "[PAUSE]" not in cleaned_transcript
        
        # Step 2: Generate audio
        result = await audio_service.generate_audio_from_transcript(
            transcript=transcript,
            video_id=video_id,
            style="tutorial"
        )
        
        # Step 3: Verify results
        assert os.path.exists(result.audio_file_path)
        assert result.duration_seconds > 0
        assert result.file_size_bytes > 0
        
        # Step 4: Generate and save metadata
        metadata = audio_processor.generate_audio_metadata(
            text=transcript,
            audio_file_path=result.audio_file_path,
            config=result.config_used.dict()
        )
        
        metadata_file = audio_processor.save_audio_metadata(metadata)
        assert os.path.exists(metadata_file)
        
        # Step 5: Load and verify metadata
        loaded_metadata = audio_processor.load_audio_metadata(metadata_file)
        assert loaded_metadata["text_metadata"]["word_count"] == len(transcript.split())
        
        print("✅ Audio service integration test completed successfully!")