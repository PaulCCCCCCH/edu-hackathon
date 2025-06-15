#!/usr/bin/env python3
"""Integration test for audio service."""

import sys
import os
import tempfile
import shutil
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

async def test_audio_service_basic():
    """Test basic audio service functionality."""
    try:
        from app.services.audio_service import AudioService, AudioConfig
        from app.utils.audio_processing import AudioProcessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create audio service
            audio_service = AudioService(storage_path=temp_dir)
            
            # Test basic audio generation
            result = await audio_service.generate_audio(
                text="Hello, this is a test of the audio generation system.",
                output_filename="test_basic.mp3"
            )
            
            assert os.path.exists(result.audio_file_path)
            assert result.duration_seconds > 0
            assert result.file_size_bytes > 0
            
            print("✅ Basic audio generation test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Basic audio generation test failed: {e}")
        return False

async def test_audio_from_transcript():
    """Test generating audio from transcript."""
    try:
        from app.services.audio_service import AudioService
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_service = AudioService(storage_path=temp_dir)
            
            transcript = "Welcome to this tutorial. [PAUSE] Let's learn about variables in Python. [PAUSE] Variables are containers for storing data values."
            
            result = await audio_service.generate_audio_from_transcript(
                transcript=transcript,
                video_id="test_tutorial_123",
                style="tutorial"
            )
            
            assert os.path.exists(result.audio_file_path)
            assert "test_tutorial_123_audio.mp3" in result.audio_file_path
            assert result.config_used.speaking_rate == 0.9  # Tutorial style
            
            print("✅ Transcript to audio test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Transcript to audio test failed: {e}")
        return False

def test_audio_processor():
    """Test audio processing utilities."""
    try:
        from app.utils.audio_processing import AudioProcessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = AudioProcessor(storage_path=temp_dir)
            
            # Test transcript cleaning
            transcript = "Hello there. [PAUSE] This is a test [PAUSE] with pauses."
            cleaned = processor.clean_transcript_for_tts(transcript)
            
            assert "[PAUSE]" not in cleaned
            assert ", " in cleaned
            
            # Test duration estimation
            duration = processor.get_audio_duration_estimate("This is a test sentence.")
            assert duration > 0
            
            # Test batch processing
            transcripts = [
                {"id": "v1", "transcript": "First video transcript"},
                {"id": "v2", "transcript": "Second video transcript"}
            ]
            
            processed = processor.batch_process_transcripts(transcripts)
            assert len(processed) == 2
            assert processed[0]["video_id"] == "v1"
            
            print("✅ Audio processor test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Audio processor test failed: {e}")
        return False

async def test_complete_workflow():
    """Test complete audio generation workflow."""
    try:
        from app.services.audio_service import AudioService
        from app.utils.audio_processing import AudioProcessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize services
            audio_service = AudioService(storage_path=temp_dir)
            processor = AudioProcessor(storage_path=temp_dir)
            
            # Sample video content
            video_content = {
                "id": "workflow_test_video",
                "title": "Introduction to Python",
                "transcript": "Hello and welcome! [PAUSE] Today we're going to learn about Python programming. [PAUSE] Python is a powerful and easy-to-learn programming language.",
                "style": "tutorial"
            }
            
            # Step 1: Process transcript
            cleaned_transcript = processor.clean_transcript_for_tts(video_content["transcript"])
            
            # Step 2: Generate audio
            audio_result = await audio_service.generate_audio_from_transcript(
                transcript=video_content["transcript"],
                video_id=video_content["id"],
                style=video_content["style"]
            )
            
            # Step 3: Generate metadata
            metadata = processor.generate_audio_metadata(
                text=video_content["transcript"],
                audio_file_path=audio_result.audio_file_path,
                config=audio_result.config_used.dict()
            )
            
            # Step 4: Save metadata
            metadata_file = processor.save_audio_metadata(metadata)
            
            # Verify everything worked
            assert os.path.exists(audio_result.audio_file_path)
            assert os.path.exists(metadata_file)
            assert audio_result.duration_seconds > 0
            
            # Load and verify metadata
            loaded_metadata = processor.load_audio_metadata(metadata_file)
            assert loaded_metadata["text_metadata"]["word_count"] > 0
            
            print("✅ Complete workflow test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        return False

async def main():
    """Run all audio integration tests."""
    print("Running integration tests for Task 2: Google TTS API Integration")
    print("=" * 70)
    
    tests = [
        test_audio_service_basic,
        test_audio_from_transcript,
        test_audio_processor,
        test_complete_workflow
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
        print("🎉 All audio integration tests passed for Task 2!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)