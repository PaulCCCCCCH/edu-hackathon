"""Service for video generation using Google Veo API."""

import os
import logging
import json
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import requests

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""
    prompt: str
    duration_seconds: float = 30.0
    resolution: str = "720p"
    style: str = "educational"
    audio_file_path: Optional[str] = None
    transcript: Optional[str] = None


class VideoGenerationResult(BaseModel):
    """Result of video generation."""
    video_file_path: str
    thumbnail_path: Optional[str] = None
    duration_seconds: float
    file_size_bytes: int
    resolution: str
    generation_method: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class VideoService:
    """Service for generating videos using Google Veo API."""
    
    def __init__(self, storage_path: str = None, api_key: str = None):
        """Initialize the video service."""
        self.storage_path = Path(storage_path or "storage/videos")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.api_key = api_key or os.getenv("GOOGLE_VEO_API_KEY")
        self.base_url = "https://veo-api.googleapis.com/v1"  # Placeholder URL
        
        if not self.api_key:
            logger.warning("Google Veo API key not available. Video generation will be mocked.")
    
    async def generate_video_from_transcript(
        self,
        transcript: str,
        video_id: str,
        audio_file_path: str = None,
        style: str = "explanation",
        duration_seconds: float = 30.0
    ) -> VideoGenerationResult:
        """Generate video from transcript and audio."""
        # Create a detailed prompt for video generation based on the transcript
        video_prompt = self._create_video_prompt(transcript, style)
        
        request = VideoGenerationRequest(
            prompt=video_prompt,
            duration_seconds=duration_seconds,
            style=style,
            audio_file_path=audio_file_path,
            transcript=transcript
        )
        
        return await self.generate_video(request, video_id)
    
    async def generate_video(
        self,
        request: VideoGenerationRequest,
        video_id: str
    ) -> VideoGenerationResult:
        """Generate video based on the request."""
        output_filename = f"{video_id}_video.mp4"
        output_path = self.storage_path / output_filename
        
        if not self.api_key:
            # Mock video generation
            return await self._mock_video_generation(request, output_path)
        
        try:
            # In a real implementation, this would call the Google Veo API
            return await self._call_veo_api(request, output_path)
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            # Fallback to mock generation
            return await self._mock_video_generation(request, output_path)
    
    async def _call_veo_api(
        self,
        request: VideoGenerationRequest,
        output_path: Path
    ) -> VideoGenerationResult:
        """Call the Google Veo API (placeholder implementation)."""
        # This is a placeholder for the actual Google Veo API call
        # In reality, this would involve:
        # 1. Authenticating with Google Cloud
        # 2. Uploading audio file if provided
        # 3. Sending generation request with prompt
        # 4. Polling for completion
        # 5. Downloading the generated video
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": request.prompt,
            "duration": request.duration_seconds,
            "resolution": request.resolution,
            "style": request.style,
            "audio_sync": request.audio_file_path is not None
        }
        
        # Simulate API call delay
        await asyncio.sleep(2)
        
        # For now, return mock generation since actual API may not be available
        logger.info("Google Veo API not yet available, using mock generation")
        return await self._mock_video_generation(request, output_path)
    
    async def _mock_video_generation(
        self,
        request: VideoGenerationRequest,
        output_path: Path
    ) -> VideoGenerationResult:
        """Mock video generation for testing and development."""
        # Create a simple mock video file
        mock_video_content = self._create_mock_video_content(request)
        
        with open(output_path, "wb") as f:
            f.write(mock_video_content)
        
        # Create thumbnail
        thumbnail_path = output_path.with_suffix(".jpg")
        mock_thumbnail = b"MOCK_THUMBNAIL_" + request.prompt.encode()[:100]
        
        with open(thumbnail_path, "wb") as f:
            f.write(mock_thumbnail)
        
        file_size = output_path.stat().st_size
        
        logger.info(f"Mock video generated: {output_path}")
        
        return VideoGenerationResult(
            video_file_path=str(output_path),
            thumbnail_path=str(thumbnail_path),
            duration_seconds=request.duration_seconds,
            file_size_bytes=file_size,
            resolution=request.resolution,
            generation_method="mock",
            metadata={
                "prompt": request.prompt,
                "style": request.style,
                "transcript_length": len(request.transcript) if request.transcript else 0,
                "has_audio": request.audio_file_path is not None
            }
        )
    
    def _create_video_prompt(self, transcript: str, style: str) -> str:
        """Create a detailed video generation prompt based on transcript and style."""
        # Extract key concepts from transcript for visual generation
        key_concepts = self._extract_visual_concepts(transcript)
        
        style_guidelines = {
            "explanation": "Clear, professional educational video with simple graphics and text overlays",
            "tutorial": "Step-by-step tutorial format with hands-on demonstrations and clear visual cues",
            "story": "Narrative-driven video with engaging visuals and smooth transitions",
            "quiz": "Interactive quiz format with questions, options, and visual feedback"
        }
        
        base_prompt = f"""Create an educational video in {style} style.
        
Content: {transcript[:500]}{'...' if len(transcript) > 500 else ''}

Visual elements to include: {', '.join(key_concepts)}

Style guidelines: {style_guidelines.get(style, 'Educational video format')}

Requirements:
- Duration: approximately {30} seconds
- Clear, engaging visuals that support the educational content
- Smooth transitions between scenes
- Text overlays for key points
- Educational and professional tone
"""
        
        return base_prompt
    
    def _extract_visual_concepts(self, transcript: str) -> List[str]:
        """Extract visual concepts from transcript for video generation."""
        # Simple keyword extraction for visual elements
        visual_keywords = [
            "python", "code", "variable", "function", "data", "algorithm",
            "graph", "chart", "diagram", "example", "step", "process",
            "concept", "idea", "theory", "practice", "tutorial", "lesson"
        ]
        
        transcript_lower = transcript.lower()
        found_concepts = []
        
        for keyword in visual_keywords:
            if keyword in transcript_lower:
                found_concepts.append(keyword)
        
        # If no specific concepts found, use general educational visuals
        if not found_concepts:
            found_concepts = ["educational graphics", "text overlay", "simple animations"]
        
        return found_concepts[:5]  # Limit to 5 concepts
    
    def _create_mock_video_content(self, request: VideoGenerationRequest) -> bytes:
        """Create mock video file content."""
        # Create a more realistic mock video file header
        # This would be a proper MP4 file in a real implementation
        
        header = b"MOCK_MP4_VIDEO_FILE"
        metadata = json.dumps({
            "prompt": request.prompt[:100],
            "duration": request.duration_seconds,
            "style": request.style,
            "generated_at": datetime.now().isoformat()
        }).encode()
        
        # Add some mock video data
        video_data = b"VIDEO_DATA_" * 1000
        
        return header + metadata + video_data
    
    async def batch_generate_videos(
        self,
        video_requests: List[Dict[str, Any]]
    ) -> List[VideoGenerationResult]:
        """Generate multiple videos in batch."""
        results = []
        
        for i, req_data in enumerate(video_requests):
            try:
                request = VideoGenerationRequest(**req_data)
                video_id = req_data.get("video_id", f"batch_video_{i}")
                
                result = await self.generate_video(request, video_id)
                results.append(result)
                
                logger.info(f"Generated video {i+1}/{len(video_requests)}: {video_id}")
                
            except Exception as e:
                logger.error(f"Error generating video {i}: {e}")
                continue
        
        logger.info(f"Batch generation completed: {len(results)}/{len(video_requests)} videos generated")
        return results
    
    def get_video_info(self, video_file_path: str) -> Dict[str, Any]:
        """Get information about a video file."""
        path = Path(video_file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_file_path}")
        
        return {
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "exists": True,
            "created_at": path.stat().st_ctime,
            "extension": path.suffix.lower()
        }
    
    def list_generated_videos(self) -> List[Dict[str, Any]]:
        """List all generated videos in the storage directory."""
        video_files = []
        
        for file_path in self.storage_path.glob("*.mp4"):
            try:
                info = self.get_video_info(str(file_path))
                video_files.append(info)
            except Exception as e:
                logger.warning(f"Could not get info for {file_path}: {e}")
        
        # Sort by creation time (newest first)
        video_files.sort(key=lambda x: x["created_at"], reverse=True)
        
        return video_files
    
    def cleanup_old_videos(self, keep_count: int = 50) -> int:
        """Clean up old video files, keeping only the most recent ones."""
        video_files = self.list_generated_videos()
        
        if len(video_files) <= keep_count:
            return 0
        
        files_to_delete = video_files[keep_count:]
        deleted_count = 0
        
        for file_info in files_to_delete:
            try:
                file_path = Path(file_info["file_path"])
                file_path.unlink()
                
                # Also delete thumbnail if it exists
                thumbnail_path = file_path.with_suffix(".jpg")
                if thumbnail_path.exists():
                    thumbnail_path.unlink()
                
                deleted_count += 1
                logger.debug(f"Deleted old video: {file_path}")
                
            except Exception as e:
                logger.warning(f"Could not delete {file_info['file_path']}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old video files")
        return deleted_count