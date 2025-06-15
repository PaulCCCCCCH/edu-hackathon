"""Service for audio generation using Gemini TTS API."""

import logging
import os
import random
import wave
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class AudioConfig(BaseModel):
    """Configuration for audio generation."""

    voice_name: str = "Kore"
    language_code: str = "en-US"
    style: str = "natural"
    model: str = "gemini-2.5-flash-preview-tts"


class AudioGenerationResult(BaseModel):
    """Result of audio generation."""

    audio_file_path: str
    duration_seconds: float | None = None
    file_size_bytes: int | None = None
    config_used: AudioConfig
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioService:
    """Service for generating audio from text using Gemini TTS API."""

    def __init__(self, storage_path: str | None = None, api_key: str | None = None) -> None:
        """Initialize the audio service."""
        self.storage_path = Path(storage_path or "../storage/audio")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found. Audio generation will be mocked.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=api_key)
                logger.info("Gemini TTS client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini TTS client: {e}. Audio generation will be mocked.")
                self.client = None

    def _save_wave_file(self, filename: str, pcm_data: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
        """Save PCM audio data as a WAV file."""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)

    def _get_wav_duration(self, filename: str) -> float:
        """Get the actual duration of a WAV file."""
        try:
            with wave.open(filename, "rb") as wf:
                frames = wf.getnframes()
                sample_rate = wf.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception as e:
            logger.warning(f"Could not get duration from WAV file {filename}: {e}")
            # Fall back to file size estimation
            file_size = Path(filename).stat().st_size
            return file_size / 48000  # Rough estimate

    async def generate_audio(
        self,
        text: str,
        output_filename: str,
        config: AudioConfig = None
    ) -> AudioGenerationResult:
        """Generate audio from text."""
        config = config or AudioConfig()

        # Ensure output filename has .wav extension
        if not output_filename.endswith(".wav"):
            output_filename = output_filename.replace(".mp3", ".wav")
            if not output_filename.endswith(".wav"):
                output_filename += ".wav"

        output_path = self.storage_path / output_filename

        if not self.client:
            # Mock audio generation for testing/development
            return await self._mock_audio_generation(text, output_path, config)

        try:
            # Prepare the prompt based on style
            prompt = self._prepare_prompt(text, config.style)

            # Generate audio using Gemini TTS
            response = self.client.models.generate_content(
                model=config.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=config.voice_name
                            )
                        )
                    )
                )
            )

            # Extract audio data from response
            if response.candidates and response.candidates[0].content.parts:
                audio_data = response.candidates[0].content.parts[0].inline_data.data

                # Save the audio data as WAV file
                self._save_wave_file(str(output_path), audio_data)
            else:
                msg = "No audio data in response"
                raise Exception(msg)

            # Calculate file size and actual duration
            file_size = output_path.stat().st_size
            actual_duration = self._get_wav_duration(str(output_path))

            logger.info(f"Audio generated successfully: {output_path} ({actual_duration:.2f}s)")

            return AudioGenerationResult(
                audio_file_path=str(output_path),
                duration_seconds=actual_duration,
                file_size_bytes=file_size,
                config_used=config,
                metadata={
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "generation_method": "gemini_tts",
                    "voice_used": config.voice_name,
                    "model_used": config.model
                }
            )

        except Exception as e:
            logger.exception(f"Error generating audio: {e}")
            # Fall back to mock generation on error
            logger.warning("Falling back to mock audio generation due to error")
            return await self._mock_audio_generation(text, output_path, config)

    def _prepare_prompt(self, text: str, style: str) -> str:
        """Prepare the prompt for TTS based on style."""
        if style == "tutorial":
            return f"Say in a clear, educational tone: {text}"
        if style == "story":
            return f"Say expressively and engagingly: {text}"
        if style == "quiz":
            return f"Say in a clear, neutral tone: {text}"
        if style == "natural":
            return f"Say naturally: {text}"
        return f"Say in a {style} voice: {text}"

    async def _mock_audio_generation(
        self,
        text: str,
        output_path: Path,
        config: AudioConfig
    ) -> AudioGenerationResult:
        """Mock audio generation for testing purposes."""
        # Estimate realistic duration based on word count
        # Average speaking rate: ~150 words per minute = 2.5 words per second
        word_count = len(text.split())
        estimated_duration = max(word_count / 2.5, 1.0)  # At least 1 second

        # Create proper WAV file with realistic duration
        sample_rate = 24000
        channels = 1
        sample_width = 2
        frames_needed = int(estimated_duration * sample_rate)

        # Generate silent audio data with proper length
        audio_data = b"\x00" * (frames_needed * channels * sample_width)

        # Save as proper WAV file
        self._save_wave_file(str(output_path), audio_data, channels, sample_rate, sample_width)

        # Get actual duration from the created file
        actual_duration = self._get_wav_duration(str(output_path))
        file_size = output_path.stat().st_size

        logger.info(f"Mock audio generated: {output_path} ({actual_duration:.2f}s)")

        return AudioGenerationResult(
            audio_file_path=str(output_path),
            duration_seconds=actual_duration,
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
        # Adjust audio config based on content style with random voice selection
        config = AudioConfig()

        # Always use random voice selection for variety
        config.voice_name = self.get_random_voice(style)

        # Clean transcript by removing pause markers
        clean_transcript = transcript.replace("[PAUSE]", ". ")

        filename = f"{video_id}_audio.wav"

        return await self.generate_audio(
            text=clean_transcript,
            output_filename=filename,
            config=config
        )

    def list_available_voices(self) -> list:
        """List available voices from Gemini TTS."""
        # Gemini TTS available voices (as per documentation)
        return [
            {"name": "Kore", "language": "en-US", "type": "professional", "description": "Clear, professional voice"},
            {"name": "Puck", "language": "en-US", "type": "expressive", "description": "Expressive, engaging voice"},
            {"name": "Aoede", "language": "en-US", "type": "neutral", "description": "Neutral, clear voice"},
            {"name": "Charon", "language": "en-US", "type": "deep", "description": "Deep, authoritative voice"},
            {"name": "Fenrir", "language": "en-US", "type": "friendly", "description": "Warm, friendly voice"},
            {"name": "Orpheus", "language": "en-US", "type": "energetic", "description": "Bright, energetic voice"}
        ]

    def get_random_voice(self, style: str | None = None) -> str:
        """Get a random voice, optionally filtered by style preference."""
        voices = self.list_available_voices()

        if style:
            # Filter voices that work well with the content style
            style_preferences = {
                "tutorial": ["professional", "neutral", "deep"],
                "story": ["expressive", "friendly", "energetic"],
                "quiz": ["neutral", "professional", "friendly"],
                "explanation": ["professional", "neutral", "expressive"]
            }

            preferred_types = style_preferences.get(style, [])
            if preferred_types:
                filtered_voices = [v for v in voices if v["type"] in preferred_types]
                if filtered_voices:
                    voices = filtered_voices

        # Select random voice
        selected_voice = random.choice(voices)
        logger.info(f"Selected random voice: {selected_voice['name']} ({selected_voice['type']}) for style: {style}")
        return selected_voice["name"]

    def get_audio_info(self, audio_file_path: str) -> dict[str, Any]:
        """Get information about an audio file."""
        path = Path(audio_file_path)

        if not path.exists():
            msg = f"Audio file not found: {audio_file_path}"
            raise FileNotFoundError(msg)

        return {
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "exists": True,
            "created_at": path.stat().st_ctime
        }
