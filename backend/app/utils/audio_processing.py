"""Utilities for audio processing and manipulation."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json


logger = logging.getLogger(__name__)


class AudioProcessor:
    """Utility class for audio processing operations."""
    
    def __init__(self, storage_path: str = None):
        """Initialize the audio processor."""
        self.storage_path = Path(storage_path or "storage/audio")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def validate_audio_file(self, file_path: str) -> bool:
        """Validate that an audio file exists and is readable."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Audio file does not exist: {file_path}")
                return False
            
            if path.stat().st_size == 0:
                logger.error(f"Audio file is empty: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio file {file_path}: {e}")
            return False
    
    def get_audio_duration_estimate(self, text: str, speaking_rate: float = 1.0) -> float:
        """Estimate audio duration based on text length and speaking rate."""
        # Rough estimation: average speaking rate is ~150 words per minute
        words = len(text.split())
        base_duration = (words / 150) * 60  # seconds
        
        # Adjust for speaking rate
        adjusted_duration = base_duration / speaking_rate
        
        # Add padding for pauses and natural speech rhythm
        padding = adjusted_duration * 0.1
        
        return adjusted_duration + padding
    
    def clean_transcript_for_tts(self, transcript: str) -> str:
        """Clean transcript text for optimal TTS processing."""
        # Remove pause markers and replace with appropriate punctuation
        cleaned = transcript.replace("[PAUSE]", ", ")
        
        # Ensure proper sentence endings
        sentences = cleaned.split(". ")
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence.endswith((".", "!", "?")):
                sentence += "."
            if sentence:
                cleaned_sentences.append(sentence)
        
        return " ".join(cleaned_sentences)
    
    def split_long_text(self, text: str, max_length: int = 5000) -> list:
        """Split long text into chunks suitable for TTS processing."""
        if len(text) <= max_length:
            return [text]
        
        # Split by sentences first
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the limit, start a new chunk
            if len(current_chunk) + len(sentence) + 2 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                current_chunk += sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_audio_metadata(
        self,
        text: str,
        audio_file_path: str,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate metadata for an audio file."""
        path = Path(audio_file_path)
        
        metadata = {
            "audio_file": str(path),
            "file_size_bytes": path.stat().st_size if path.exists() else 0,
            "text_metadata": {
                "original_text": text,
                "cleaned_text": self.clean_transcript_for_tts(text),
                "word_count": len(text.split()),
                "character_count": len(text),
                "estimated_duration_seconds": self.get_audio_duration_estimate(text)
            },
            "processing_config": config or {},
            "created_at": path.stat().st_ctime if path.exists() else None
        }
        
        return metadata
    
    def save_audio_metadata(
        self,
        metadata: Dict[str, Any],
        metadata_file_path: str = None
    ) -> str:
        """Save audio metadata to a JSON file."""
        if not metadata_file_path:
            audio_path = Path(metadata["audio_file"])
            metadata_file_path = audio_path.with_suffix(".json")
        
        with open(metadata_file_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Audio metadata saved to: {metadata_file_path}")
        return str(metadata_file_path)
    
    def load_audio_metadata(self, metadata_file_path: str) -> Dict[str, Any]:
        """Load audio metadata from a JSON file."""
        try:
            with open(metadata_file_path, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Error loading audio metadata from {metadata_file_path}: {e}")
            return {}
    
    def cleanup_temp_files(self, keep_patterns: list = None) -> int:
        """Clean up temporary audio files."""
        keep_patterns = keep_patterns or ["*.mp3", "*.json"]
        
        temp_dir = self.storage_path / "temp"
        if not temp_dir.exists():
            return 0
        
        cleaned_count = 0
        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                # Check if file matches any keep pattern
                should_keep = any(
                    file_path.match(pattern) for pattern in keep_patterns
                )
                
                if not should_keep:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Cleaned up temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not clean up {file_path}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary audio files")
        return cleaned_count
    
    def batch_process_transcripts(
        self,
        transcripts: list,
        output_dir: str = None
    ) -> list:
        """Process multiple transcripts and prepare them for audio generation."""
        output_dir = Path(output_dir or self.storage_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_transcripts = []
        
        for i, transcript in enumerate(transcripts):
            if isinstance(transcript, dict):
                text = transcript.get("transcript", "")
                video_id = transcript.get("id", f"video_{i}")
            else:
                text = str(transcript)
                video_id = f"video_{i}"
            
            # Clean the transcript
            cleaned_text = self.clean_transcript_for_tts(text)
            
            # Split if too long
            chunks = self.split_long_text(cleaned_text)
            
            processed_transcript = {
                "video_id": video_id,
                "original_text": text,
                "cleaned_text": cleaned_text,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "estimated_duration": self.get_audio_duration_estimate(cleaned_text),
                "output_file": f"{video_id}_audio.mp3"
            }
            
            processed_transcripts.append(processed_transcript)
        
        logger.info(f"Processed {len(processed_transcripts)} transcripts for audio generation")
        return processed_transcripts