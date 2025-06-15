"""Service for audio generation using Google Text-to-Speech API."""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json

try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False
    texttospeech = None

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class AudioConfig(BaseModel):
    """Configuration for audio generation."""
    voice_name: str = "en-US-Wavenet-D"
    language_code: str = "en-US"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    audio_encoding: str = "MP3"
    sample_rate_hertz: int = 24000


class AudioGenerationResult(BaseModel):
    """Result of audio generation."""
    audio_file_path: str
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    config_used: AudioConfig
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AudioService:
    """Service for generating audio from text using Google TTS API."""
    
    def __init__(self, storage_path: str = None, credentials_path: str = None):
        """Initialize the audio service."""
        self.storage_path = Path(storage_path or "storage/audio")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        if not GOOGLE_TTS_AVAILABLE:
            logger.warning("Google Cloud Text-to-Speech library not available. Audio generation will be mocked.")
            self.client = None
        else:
            try:
                if credentials_path:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                self.client = texttospeech.TextToSpeechClient()
                logger.info("Google TTS client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google TTS client: {e}. Audio generation will be mocked.")
                self.client = None
    
    async def generate_audio(
        self,
        text: str,
        output_filename: str,
        config: AudioConfig = None
    ) -> AudioGenerationResult:
        """Generate audio from text."""
        config = config or AudioConfig()
        
        output_path = self.storage_path / output_filename
        
        if not self.client:
            # Mock audio generation for testing/development
            return await self._mock_audio_generation(text, output_path, config)
        
        try:
            # Configure the synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=config.language_code,
                name=config.voice_name
            )
            
            # Select the type of audio file
            audio_config = texttospeech.AudioConfig(
                audio_encoding=getattr(texttospeech.AudioEncoding, config.audio_encoding),
                speaking_rate=config.speaking_rate,
                pitch=config.pitch,
                sample_rate_hertz=config.sample_rate_hertz
            )
            
            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Write the response to the output file
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
            
            # Calculate file size
            file_size = output_path.stat().st_size
            
            # Estimate duration (rough approximation)
            estimated_duration = len(text.split()) * 0.6  # ~0.6 seconds per word
            
            logger.info(f"Audio generated successfully: {output_path}")
            
            return AudioGenerationResult(
                audio_file_path=str(output_path),
                duration_seconds=estimated_duration,
                file_size_bytes=file_size,
                config_used=config,
                metadata={
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "generation_method": "google_tts"
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise
    
    async def _mock_audio_generation(
        self,
        text: str,
        output_path: Path,
        config: AudioConfig
    ) -> AudioGenerationResult:
        """Mock audio generation for testing purposes."""
        # Create a dummy audio file
        mock_audio_content = b"MOCK_AUDIO_DATA_" + text.encode()[:100]
        
        with open(output_path, "wb") as f:
            f.write(mock_audio_content)
        
        estimated_duration = len(text.split()) * 0.6
        file_size = len(mock_audio_content)
        
        logger.info(f"Mock audio generated: {output_path}")
        
        return AudioGenerationResult(
            audio_file_path=str(output_path),
            duration_seconds=estimated_duration,
            file_size_bytes=file_size,
            config_used=config,
            metadata={
                "text_length": len(text),
                "word_count": len(text.split()),
                "generation_method": "mock"
            }
        )
    
    async def generate_audio_from_transcript(
        self,
        transcript: str,
        video_id: str,
        style: str = "explanation"
    ) -> AudioGenerationResult:
        """Generate audio from a video transcript with style-appropriate settings."""
        # Adjust audio config based on content style
        config = AudioConfig()
        
        if style == "tutorial":
            config.speaking_rate = 0.9  # Slightly slower for tutorials
            config.voice_name = "en-US-Wavenet-A"  # More formal voice
        elif style == "story":
            config.speaking_rate = 1.1  # Slightly faster for stories
            config.voice_name = "en-US-Wavenet-F"  # More expressive voice
        elif style == "quiz":
            config.speaking_rate = 1.0
            config.voice_name = "en-US-Wavenet-D"  # Clear, neutral voice
        
        # Clean transcript by removing pause markers
        clean_transcript = transcript.replace("[PAUSE]", ". ")
        
        filename = f"{video_id}_audio.mp3"
        
        return await self.generate_audio(
            text=clean_transcript,
            output_filename=filename,
            config=config
        )
    
    def list_available_voices(self) -> list:
        """List available voices from Google TTS."""
        if not self.client:
            return [
                {"name": "en-US-Wavenet-A", "language": "en-US", "type": "mock"},
                {"name": "en-US-Wavenet-D", "language": "en-US", "type": "mock"},
                {"name": "en-US-Wavenet-F", "language": "en-US", "type": "mock"}
            ]
        
        try:
            voices = self.client.list_voices()
            return [
                {
                    "name": voice.name,
                    "language": voice.language_codes[0] if voice.language_codes else "unknown",
                    "gender": voice.ssml_gender.name
                }
                for voice in voices.voices
                if voice.language_codes and voice.language_codes[0].startswith("en")
            ]
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []
    
    def get_audio_info(self, audio_file_path: str) -> Dict[str, Any]:
        """Get information about an audio file."""
        path = Path(audio_file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        return {
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "exists": True,
            "created_at": path.stat().st_ctime
        }