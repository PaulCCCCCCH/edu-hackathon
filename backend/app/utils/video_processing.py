"""Utilities for video processing and manipulation."""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Utility class for video processing operations."""
    
    def __init__(self, storage_path: str = None):
        """Initialize the video processor."""
        self.storage_path = Path(storage_path or "storage/videos")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def validate_video_file(self, file_path: str) -> bool:
        """Validate that a video file exists and has content."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Video file does not exist: {file_path}")
                return False
            
            if path.stat().st_size == 0:
                logger.error(f"Video file is empty: {file_path}")
                return False
            
            # Check if it has a video extension
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            if path.suffix.lower() not in video_extensions:
                logger.warning(f"File may not be a video: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating video file {file_path}: {e}")
            return False
    
    def extract_video_metadata(self, video_file_path: str) -> Dict[str, Any]:
        """Extract metadata from a video file."""
        path = Path(video_file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_file_path}")
        
        # Basic file information
        stat = path.stat()
        
        metadata = {
            "file_path": str(path),
            "filename": path.name,
            "file_size_bytes": stat.st_size,
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime,
            "extension": path.suffix.lower(),
            "estimated_duration": self._estimate_video_duration(path),
            "file_type": "video"
        }
        
        # Try to extract additional metadata if available
        try:
            metadata.update(self._extract_advanced_metadata(path))
        except Exception as e:
            logger.debug(f"Could not extract advanced metadata: {e}")
        
        return metadata
    
    def _estimate_video_duration(self, video_path: Path) -> float:
        """Estimate video duration based on file size (rough approximation)."""
        # Very rough estimation: assume ~1MB per second for educational videos
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        estimated_duration = max(file_size_mb, 10.0)  # Minimum 10 seconds
        return min(estimated_duration, 300.0)  # Maximum 5 minutes
    
    def _extract_advanced_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract advanced metadata if video processing tools are available."""
        # This would use tools like ffmpeg or similar in a real implementation
        # For now, return basic information
        
        return {
            "resolution": "unknown",
            "fps": "unknown",
            "codec": "unknown",
            "bitrate": "unknown"
        }
    
    def create_video_thumbnail(
        self,
        video_file_path: str,
        thumbnail_path: str = None,
        timestamp: float = 1.0
    ) -> str:
        """Create a thumbnail from a video file."""
        video_path = Path(video_file_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_file_path}")
        
        if not thumbnail_path:
            thumbnail_path = video_path.with_suffix(".jpg")
        else:
            thumbnail_path = Path(thumbnail_path)
        
        # Mock thumbnail creation
        # In a real implementation, this would use ffmpeg or similar
        thumbnail_content = b"MOCK_THUMBNAIL_" + video_path.name.encode()
        
        with open(thumbnail_path, "wb") as f:
            f.write(thumbnail_content)
        
        logger.info(f"Created thumbnail: {thumbnail_path}")
        return str(thumbnail_path)
    
    def generate_video_manifest(
        self,
        video_file_path: str,
        additional_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive manifest for a video file."""
        metadata = self.extract_video_metadata(video_file_path)
        
        manifest = {
            "video_info": metadata,
            "manifest_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "additional_info": additional_info or {}
        }
        
        # Add thumbnail if it exists
        video_path = Path(video_file_path)
        thumbnail_path = video_path.with_suffix(".jpg")
        if thumbnail_path.exists():
            manifest["thumbnail_path"] = str(thumbnail_path)
        
        return manifest
    
    def save_video_manifest(
        self,
        manifest: Dict[str, Any],
        manifest_file_path: str = None
    ) -> str:
        """Save video manifest to a JSON file."""
        if not manifest_file_path:
            video_path = Path(manifest["video_info"]["file_path"])
            manifest_file_path = video_path.with_suffix(".manifest.json")
        
        with open(manifest_file_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        
        logger.info(f"Video manifest saved to: {manifest_file_path}")
        return str(manifest_file_path)
    
    def load_video_manifest(self, manifest_file_path: str) -> Dict[str, Any]:
        """Load video manifest from a JSON file."""
        try:
            with open(manifest_file_path, "r") as f:
                manifest = json.load(f)
            return manifest
        except Exception as e:
            logger.error(f"Error loading video manifest from {manifest_file_path}: {e}")
            return {}
    
    def batch_process_videos(
        self,
        video_files: List[str],
        generate_thumbnails: bool = True,
        generate_manifests: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple video files in batch."""
        results = []
        
        for video_file in video_files:
            try:
                if not self.validate_video_file(video_file):
                    logger.warning(f"Skipping invalid video file: {video_file}")
                    continue
                
                result = {
                    "video_file": video_file,
                    "processed": True,
                    "thumbnail_path": None,
                    "manifest_path": None,
                    "metadata": None
                }
                
                # Extract metadata
                result["metadata"] = self.extract_video_metadata(video_file)
                
                # Generate thumbnail
                if generate_thumbnails:
                    try:
                        result["thumbnail_path"] = self.create_video_thumbnail(video_file)
                    except Exception as e:
                        logger.warning(f"Could not create thumbnail for {video_file}: {e}")
                
                # Generate manifest
                if generate_manifests:
                    try:
                        manifest = self.generate_video_manifest(video_file)
                        result["manifest_path"] = self.save_video_manifest(manifest)
                    except Exception as e:
                        logger.warning(f"Could not create manifest for {video_file}: {e}")
                
                results.append(result)
                logger.info(f"Processed video: {video_file}")
                
            except Exception as e:
                logger.error(f"Error processing video {video_file}: {e}")
                results.append({
                    "video_file": video_file,
                    "processed": False,
                    "error": str(e)
                })
        
        logger.info(f"Batch processing completed: {len(results)} videos processed")
        return results
    
    def organize_videos_by_topic(
        self,
        video_manifests: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Organize videos by topic based on their metadata."""
        topics = {}
        
        for manifest in video_manifests:
            additional_info = manifest.get("additional_info", {})
            video_topics = additional_info.get("topics", ["uncategorized"])
            
            if isinstance(video_topics, str):
                video_topics = [video_topics]
            
            for topic in video_topics:
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append(manifest)
        
        return topics
    
    def get_video_statistics(self, video_directory: str = None) -> Dict[str, Any]:
        """Get statistics about videos in the storage directory."""
        video_dir = Path(video_directory or self.storage_path)
        
        if not video_dir.exists():
            return {"error": "Directory does not exist"}
        
        video_files = list(video_dir.glob("*.mp4"))
        
        total_size = sum(f.stat().st_size for f in video_files)
        total_count = len(video_files)
        
        # Calculate size distribution
        sizes = [f.stat().st_size for f in video_files]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        
        stats = {
            "total_videos": total_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "average_size_bytes": avg_size,
            "average_size_mb": avg_size / (1024 * 1024),
            "storage_directory": str(video_dir)
        }
        
        return stats
    
    def cleanup_incomplete_files(self) -> int:
        """Clean up incomplete or corrupted video files."""
        cleaned_count = 0
        
        for video_file in self.storage_path.glob("*.mp4"):
            try:
                # Check for very small files (likely incomplete)
                if video_file.stat().st_size < 1024:  # Less than 1KB
                    video_file.unlink()
                    cleaned_count += 1
                    logger.info(f"Removed incomplete file: {video_file}")
                    
                    # Also remove associated files
                    thumbnail = video_file.with_suffix(".jpg")
                    if thumbnail.exists():
                        thumbnail.unlink()
                    
                    manifest = video_file.with_suffix(".manifest.json")
                    if manifest.exists():
                        manifest.unlink()
                        
            except Exception as e:
                logger.warning(f"Could not check/clean file {video_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} incomplete video files")
        return cleaned_count