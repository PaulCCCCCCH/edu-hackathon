#!/usr/bin/env python3
"""Basic test to verify the content generator structure."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_content_generator_structure():
    """Test that the content generator can be imported and has required methods."""
    try:
        # Mock the openai import since it's not installed
        import unittest.mock
        with unittest.mock.patch.dict('sys.modules', {'openai': unittest.mock.MagicMock()}):
            from app.services.content_generator import ContentGeneratorService, VideoContent, VideoConcept
            
            # Test that we can create instances
            service = ContentGeneratorService()
            assert service is not None
            assert hasattr(service, 'generate_video_transcript')
            assert hasattr(service, 'generate_video_batch')
            assert hasattr(service, 'recommend_content')
            
            # Test VideoContent model
            video_content = VideoContent(
                title="Test Video",
                transcript="Test transcript",
                topics=["test"],
                difficulty_level="beginner",
                duration_seconds=30.0,
                style="explanation"
            )
            assert video_content.title == "Test Video"
            assert video_content.topics == ["test"]
            
            # Test VideoConcept model
            concept = VideoConcept(
                topic="Test Topic",
                difficulty_level="beginner",
                style="explanation",
                target_audience="students",
                connection_to_interests="test connection"
            )
            assert concept.topic == "Test Topic"
            
            print("✅ Content generator structure test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Content generator structure test failed: {e}")
        return False

def test_schemas_exist():
    """Test that the required schema files exist."""
    schema_path = os.path.join(os.path.dirname(__file__), 'app', 'schemas', 'video_schema.json')
    if os.path.exists(schema_path):
        print("✅ Video schema file exists!")
        return True
    else:
        print("❌ Video schema file not found!")
        return False

def test_prompts_exist():
    """Test that the prompt files exist."""
    prompts_path = os.path.join(os.path.dirname(__file__), 'app', 'prompts', 'video_generation_prompts.py')
    if os.path.exists(prompts_path):
        print("✅ Video generation prompts file exists!")
        return True
    else:
        print("❌ Video generation prompts file not found!")
        return False

if __name__ == "__main__":
    print("Running basic integration tests for Task 1: GPT-4o Integration")
    print("=" * 60)
    
    tests = [
        test_schemas_exist,
        test_prompts_exist,
        test_content_generator_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All basic integration tests passed for Task 1!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    sys.exit(0 if passed == total else 1)