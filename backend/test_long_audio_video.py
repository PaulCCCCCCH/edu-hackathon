#!/usr/bin/env python3
"""Test script to verify that short videos are properly looped to match longer audio."""

import asyncio
import logging
from pathlib import Path

from app.services.audio_service import AudioService
from app.services.video_service import VideoService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_long_audio_video_looping():
    """Test that a short video is looped to match a longer audio duration."""
    
    # Create a longer transcript to generate longer audio
    long_transcript = """
    Machine learning is a fascinating field that combines computer science, statistics, and domain expertise.
    It involves teaching computers to learn patterns from data without being explicitly programmed for every scenario.
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
    Supervised learning uses labeled data to train models that can make predictions on new, unseen data.
    Unsupervised learning finds hidden patterns in data without labeled examples.
    Reinforcement learning trains agents to make decisions by learning from rewards and punishments.
    Popular algorithms include linear regression, decision trees, neural networks, and support vector machines.
    The key to success in machine learning is having quality data, choosing the right algorithm, and proper evaluation.
    Applications range from recommendation systems to autonomous vehicles to medical diagnosis.
    As the field continues to evolve, it promises to transform many industries and aspects of our daily lives.
    """
    
    # Initialize services
    audio_service = AudioService(storage_path="storage/audio")
    video_service = VideoService(storage_path="storage/videos")
    
    video_id = "test_long_audio"
    
    logger.info("=== Testing Long Audio with Short Video Looping ===")
    
    # Step 1: Generate audio (should be longer than 5 seconds)
    logger.info("Generating audio from long transcript...")
    audio_result = await audio_service.generate_audio_from_transcript(
        transcript=long_transcript,
        video_id=video_id,
        style="explanation"
    )
    
    logger.info(f"Audio generated: {audio_result.duration_seconds:.2f}s")
    
    # Step 2: Generate video (should loop to match audio duration)
    logger.info("Generating video with looping to match audio duration...")
    video_result = await video_service.generate_video_from_transcript(
        transcript=long_transcript,
        video_id=video_id,
        audio_file_path=audio_result.audio_file_path,
        style="explanation",
        duration_seconds=audio_result.duration_seconds
    )
    
    logger.info(f"Video generated: {video_result.duration_seconds:.2f}s")
    
    # Step 3: Verify durations match
    audio_path = Path(audio_result.audio_file_path)
    video_path = Path(video_result.video_file_path)
    
    logger.info(f"Audio file: {audio_path} (exists: {audio_path.exists()})")
    logger.info(f"Video file: {video_path} (exists: {video_path.exists()})")
    
    if audio_path.exists() and video_path.exists():
        # Check actual file durations
        import subprocess
        
        # Get audio duration
        audio_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_path)]
        audio_duration_result = subprocess.run(audio_cmd, capture_output=True, text=True)
        actual_audio_duration = float(audio_duration_result.stdout.strip()) if audio_duration_result.returncode == 0 else 0
        
        # Get video duration
        video_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)]
        video_duration_result = subprocess.run(video_cmd, capture_output=True, text=True)
        actual_video_duration = float(video_duration_result.stdout.strip()) if video_duration_result.returncode == 0 else 0
        
        logger.info(f"Actual audio duration: {actual_audio_duration:.2f}s")
        logger.info(f"Actual video duration: {actual_video_duration:.2f}s")
        
        # Verify they match (within 0.1 second tolerance)
        duration_diff = abs(actual_audio_duration - actual_video_duration)
        if duration_diff <= 0.1:
            logger.info("✅ SUCCESS: Video duration matches audio duration!")
            logger.info(f"   Audio: {actual_audio_duration:.2f}s")
            logger.info(f"   Video: {actual_video_duration:.2f}s")
            logger.info(f"   Difference: {duration_diff:.3f}s (within tolerance)")
            
            # Check if looping occurred
            if actual_audio_duration > 10:  # If audio is longer than 10 seconds
                logger.info("✅ SUCCESS: Audio is longer than 10s, video should have been looped!")
                logger.info(f"   Original video duration would be ~5s (SHORT_VIDEO_DURATION)")
                logger.info(f"   Final video duration: {actual_video_duration:.2f}s")
                loops_estimated = int(actual_video_duration / 5) 
                logger.info(f"   Estimated loops needed: {loops_estimated}")
            
        else:
            logger.error("❌ FAILURE: Video duration does not match audio duration!")
            logger.error(f"   Audio: {actual_audio_duration:.2f}s")
            logger.error(f"   Video: {actual_video_duration:.2f}s")
            logger.error(f"   Difference: {duration_diff:.3f}s (exceeds tolerance)")
    else:
        logger.error("❌ FAILURE: Generated files not found!")
    
    logger.info("=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_long_audio_video_looping())