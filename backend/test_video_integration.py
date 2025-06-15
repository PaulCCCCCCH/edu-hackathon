#!/usr/bin/env python3
"""Integration test for video service."""

import sys
import os
import tempfile
import shutil
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

async def test_video_service_basic():
    """Test basic video service functionality."""
    try:
        from app.services.video_service import VideoService, VideoGenerationRequest
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create video service
            video_service = VideoService(storage_path=temp_dir)
            
            # Test basic video generation
            request = VideoGenerationRequest(
                prompt="Create an educational video about Python variables",
                duration_seconds=30.0,
                style="tutorial"
            )
            
            result = await video_service.generate_video(request, "test_video_123")
            
            assert os.path.exists(result.video_file_path)
            assert result.duration_seconds == 30.0
            assert result.file_size_bytes > 0
            assert result.generation_method == "mock"
            assert result.thumbnail_path and os.path.exists(result.thumbnail_path)
            
            print("✅ Basic video generation test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Basic video generation test failed: {e}")
        return False

async def test_video_from_transcript():
    """Test generating video from transcript."""
    try:
        from app.services.video_service import VideoService
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_service = VideoService(storage_path=temp_dir)
            
            transcript = "Welcome to this Python tutorial. Today we'll learn about variables. Variables in Python are containers for storing data values."
            
            result = await video_service.generate_video_from_transcript(
                transcript=transcript,
                video_id="transcript_test_video",
                style="tutorial"
            )
            
            assert os.path.exists(result.video_file_path)
            assert "transcript_test_video_video.mp4" in result.video_file_path
            assert result.metadata["transcript_length"] == len(transcript)
            assert result.metadata["style"] == "tutorial"
            
            print("✅ Transcript to video test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Transcript to video test failed: {e}")
        return False

def test_video_processor():
    """Test video processing utilities."""
    try:
        from app.utils.video_processing import VideoProcessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VideoProcessor(storage_path=temp_dir)
            
            # Create a mock video file
            video_file = os.path.join(temp_dir, "test_video.mp4")
            with open(video_file, "wb") as f:
                f.write(b"MOCK_VIDEO_DATA" * 100)
            
            # Test video validation
            assert processor.validate_video_file(video_file)
            
            # Test metadata extraction
            metadata = processor.extract_video_metadata(video_file)
            assert metadata["file_size_bytes"] > 0
            assert metadata["filename"] == "test_video.mp4"
            assert metadata["extension"] == ".mp4"
            
            # Test thumbnail creation
            thumbnail_path = processor.create_video_thumbnail(video_file)
            assert os.path.exists(thumbnail_path)
            
            # Test manifest generation
            manifest = processor.generate_video_manifest(video_file)
            assert "video_info" in manifest
            assert "generated_at" in manifest
            
            # Test manifest saving/loading
            manifest_path = processor.save_video_manifest(manifest)
            assert os.path.exists(manifest_path)
            
            loaded_manifest = processor.load_video_manifest(manifest_path)
            assert loaded_manifest["video_info"]["filename"] == "test_video.mp4"
            
            print("✅ Video processor test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Video processor test failed: {e}")
        return False

async def test_batch_video_generation():
    """Test batch video generation."""
    try:
        from app.services.video_service import VideoService
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_service = VideoService(storage_path=temp_dir)
            
            # Create batch requests
            requests = [
                {
                    "video_id": "batch_video_1",
                    "prompt": "Video about Python basics",
                    "duration_seconds": 25.0,
                    "style": "explanation"
                },
                {
                    "video_id": "batch_video_2",
                    "prompt": "Video about Python functions",
                    "duration_seconds": 35.0,
                    "style": "tutorial"
                }
            ]
            
            results = await video_service.batch_generate_videos(requests)
            
            assert len(results) == 2
            assert all(os.path.exists(result.video_file_path) for result in results)
            assert results[0].duration_seconds == 25.0
            assert results[1].duration_seconds == 35.0
            
            print("✅ Batch video generation test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Batch video generation test failed: {e}")
        return False

async def test_complete_video_workflow():
    """Test complete video generation workflow."""
    try:
        from app.services.video_service import VideoService
        from app.utils.video_processing import VideoProcessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize services
            video_service = VideoService(storage_path=temp_dir)
            processor = VideoProcessor(storage_path=temp_dir)
            
            # Sample content
            transcript = "Welcome to Python programming! In this lesson, we'll explore variables. Variables are fundamental building blocks in Python."
            
            # Step 1: Generate video from transcript
            video_result = await video_service.generate_video_from_transcript(
                transcript=transcript,
                video_id="workflow_test_video",
                style="explanation"
            )
            
            # Step 2: Process video (create thumbnail, manifest)
            processor_results = processor.batch_process_videos(
                [video_result.video_file_path],
                generate_thumbnails=True,
                generate_manifests=True
            )
            
            # Step 3: Verify everything worked
            assert os.path.exists(video_result.video_file_path)
            assert len(processor_results) == 1
            assert processor_results[0]["processed"] is True
            assert processor_results[0]["thumbnail_path"] is not None
            assert processor_results[0]["manifest_path"] is not None
            
            # Step 4: Load and verify manifest
            manifest = processor.load_video_manifest(processor_results[0]["manifest_path"])
            assert manifest["video_info"]["filename"].endswith(".mp4")
            
            # Step 5: Get video statistics
            stats = processor.get_video_statistics(temp_dir)
            assert stats["total_videos"] == 1
            assert stats["total_size_bytes"] > 0
            
            print("✅ Complete video workflow test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Complete video workflow test failed: {e}")
        return False

def test_video_organization():
    """Test video organization features."""
    try:
        from app.utils.video_processing import VideoProcessor
        
        processor = VideoProcessor()
        
        # Create sample manifests
        manifests = [
            {
                "video_info": {"filename": "python_basics.mp4"},
                "additional_info": {"topics": ["python", "programming"]}
            },
            {
                "video_info": {"filename": "math_tutorial.mp4"},
                "additional_info": {"topics": ["mathematics", "algebra"]}
            },
            {
                "video_info": {"filename": "python_advanced.mp4"},
                "additional_info": {"topics": ["python", "advanced"]}
            }
        ]
        
        # Test organization by topic
        organized = processor.organize_videos_by_topic(manifests)
        
        assert "python" in organized
        assert len(organized["python"]) == 2
        assert "mathematics" in organized
        assert len(organized["mathematics"]) == 1
        
        print("✅ Video organization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Video organization test failed: {e}")
        return False

async def main():
    """Run all video integration tests."""
    print("Running integration tests for Task 3: Google Veo API Integration")
    print("=" * 70)
    
    tests = [
        test_video_service_basic,
        test_video_from_transcript,
        test_video_processor,
        test_batch_video_generation,
        test_complete_video_workflow,
        test_video_organization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if asyncio.iscoroutinefunction(test):
            if await test():
                passed += 1
        else:
            if test():
                passed += 1
    
    print("=" * 70)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All video integration tests passed for Task 3!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)