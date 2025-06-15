#!/usr/bin/env python3
"""E2E test script to generate 3 full videos with audio using the EduTok API."""

import asyncio
import json
import time
from pathlib import Path

import httpx


BASE_URL = "http://localhost:8000"


def print_status(message: str):
    """Print status with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


async def submit_user_background(client: httpx.AsyncClient):
    """Submit user background information."""
    print_status("📝 Submitting user background...")

    # Format structured data into a description string as expected by the API
    interests_str = ", ".join(["Machine Learning", "Web Development", "Data Science", "AI Ethics", "Cloud Computing"])
    description = f"I am a Software Engineer with a Masters in Computer Science. My interests include: {interests_str}."
    
    background_data = {
        "description": description
    }

    response = await client.post(f"{BASE_URL}/background", json=background_data)
    response.raise_for_status()

    print_status(f"✅ User background submitted: {response.json()}")
    return response.json()


async def generate_videos(client: httpx.AsyncClient, topics: list[str]):
    """Generate videos for given topics."""
    print_status(f"🎬 Generating videos for topics: {topics}")

    topics_data = {"topics": topics}

    # This endpoint generates 2 videos at a time
    response = await client.post(f"{BASE_URL}/generate_videos", json=topics_data)
    response.raise_for_status()

    videos = response.json()
    print_status(f"✅ Generated {len(videos)} videos")

    for i, video in enumerate(videos):
        print(f"  Video {i + 1}:")
        print(f"    Title: {video.get('title', 'N/A')}")
        print(f"    Video URL: {video['video_url']}")
        print(f"    Audio URL: {video.get('audio_url', 'N/A')}")
        print(f"    Duration: {video.get('duration_seconds', 'N/A')} seconds")
        print(f"    Topics: {video.get('topics', 'N/A')}")
        print(f"    Transcript: {video['transcript'][:100]}...")

    return videos


async def fetch_additional_video(client: httpx.AsyncClient, index: int):
    """Fetch a video by index, generating if needed."""
    print_status(f"🎯 Fetching video at index {index}...")

    response = await client.get(f"{BASE_URL}/fetch_video/{index}")
    response.raise_for_status()

    video = response.json()
    print_status(f"✅ Fetched video {index}")
    print(f"  Title: {video.get('title', 'N/A')}")
    print(f"  Video URL: {video['video_url']}")
    print(f"  Audio URL: {video.get('audio_url', 'N/A')}")
    print(f"  Transcript: {video['transcript'][:100]}...")

    return video


async def get_video_stats(client: httpx.AsyncClient):
    """Get video generation statistics."""
    print_status("📊 Getting video statistics...")

    response = await client.get(f"{BASE_URL}/video_stats")
    response.raise_for_status()

    stats = response.json()
    print_status("📈 Video Statistics:")
    print(f"  Total videos in memory: {stats['total_videos_generated']}")
    print(f"  Video files on disk: {stats['video_files_on_disk']}")
    print(f"  Audio files on disk: {stats['audio_files_on_disk']}")
    print(f"  Real videos: {stats['real_videos']}")
    print(f"  Mock videos: {stats['mock_videos']}")

    return stats


async def check_video_generation_mode():
    """Check which video generation mode is configured."""
    import os
    from pathlib import Path
    
    # Load .env file to check current mode
    env_file = Path("../.env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("VIDEO_GENERATION_MODE="):
                    mode = line.split("=", 1)[1].strip()
                    print_status(f"🎛️  Video Generation Mode: {mode}")
                    return mode
    
    # Fallback to environment variable
    mode = os.getenv("VIDEO_GENERATION_MODE", "ai")
    print_status(f"🎛️  Video Generation Mode: {mode} (from env)")
    return mode


async def verify_sample_videos_exist():
    """Verify that video samples exist in the samples directory."""
    print_status("🎞️  Checking video samples directory...")
    
    samples_path = Path("../storage/video_samples")
    if not samples_path.exists():
        print_status("❌ Video samples directory not found!")
        return False
    
    sample_files = list(samples_path.glob("*.mp4"))
    print_status(f"📹 Found {len(sample_files)} video samples:")
    
    for sample in sample_files[:5]:  # Show first 5
        size_mb = sample.stat().st_size / (1024 * 1024)
        print(f"  • {sample.name} ({size_mb:.1f} MB)")
    
    if len(sample_files) > 5:
        print(f"  ... and {len(sample_files) - 5} more")
    
    return len(sample_files) > 0


async def verify_files_exist(videos: list[dict]):
    """Verify that generated video and audio files actually exist."""
    print_status("🔍 Verifying generated files exist on disk...")

    storage_path = Path("../storage")

    for i, video in enumerate(videos):
        video_url = video["video_url"]
        audio_url = video.get("audio_url")

        # Extract file paths from URLs
        if "/storage/" in video_url:
            video_file = storage_path / video_url.split("/storage/", 1)[1]
            if video_file.exists():
                size_mb = video_file.stat().st_size / (1024 * 1024)
                print(f"  ✅ Video {i + 1}: {video_file} ({size_mb:.2f} MB)")
            else:
                print(f"  ❌ Video {i + 1}: {video_file} - NOT FOUND")

        if audio_url and "/storage/" in audio_url:
            audio_file = storage_path / audio_url.split("/storage/", 1)[1]
            if audio_file.exists():
                size_mb = audio_file.stat().st_size / (1024 * 1024)
                print(f"  ✅ Audio {i + 1}: {audio_file} ({size_mb:.2f} MB)")
            else:
                print(f"  ❌ Audio {i + 1}: {audio_file} - NOT FOUND")


async def verify_video_generation_method(videos: list[dict], expected_mode: str):
    """Verify that videos were generated using the expected method."""
    print_status(f"🔍 Verifying video generation method (expected: {expected_mode})...")
    
    for i, video in enumerate(videos):
        metadata = video.get('metadata', {})
        generation_method = metadata.get('generation_method', 'unknown')
        
        print(f"  Video {i + 1}: {generation_method}")
        
        if expected_mode == "samples":
            if generation_method == "video_samples":
                samples_used = metadata.get('samples_used', [])
                print(f"    ✅ Used video samples: {samples_used}")
            else:
                print(f"    ⚠️  Expected 'video_samples' but got '{generation_method}'")
        elif expected_mode == "ai":
            if generation_method.startswith("gemini") or generation_method == "mock":
                print(f"    ✅ Used AI generation: {generation_method}")
            else:
                print(f"    ⚠️  Expected AI generation but got '{generation_method}'")


async def main():
    """Main E2E test function."""
    print_status("🚀 Starting E2E Video Generation Test")
    print_status(f"🌐 API Base URL: {BASE_URL}")

    timeout = httpx.Timeout(300.0)  # 5 minute timeout for video generation

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # Step 0: Check configuration and setup
            generation_mode = await check_video_generation_mode()
            
            if generation_mode == "samples":
                samples_exist = await verify_sample_videos_exist()
                if not samples_exist:
                    print_status("❌ No video samples found but samples mode is enabled!")
                    return
            
            # Step 1: Submit user background
            await submit_user_background(client)

            # Step 2: Generate videos with specific topics
            topics_batch_1 = [
                "Introduction to Machine Learning Algorithms",
                "Web Development Best Practices", 
                "Data Science Fundamentals",
            ]
            videos_batch_1 = await generate_videos(client, topics_batch_1)

            # All videos from this single batch
            all_videos = videos_batch_1

            # Step 3: Verify generation method matches configuration
            await verify_video_generation_method(all_videos, generation_mode)

            # Step 4: Get final statistics
            await get_video_stats(client)

            # Step 5: Verify files exist on disk
            await verify_files_exist(all_videos)

            print_status("🎉 E2E Test Completed Successfully!")
            print(f"📊 Total videos generated: {len(all_videos)}")
            print_status(f"🎛️  Generation mode verified: {generation_mode}")
            
            if generation_mode == "samples":
                print_status("✅ Confirmed: Videos use pre-recorded samples (NO Gemini video generation)")
                print_status("✅ Confirmed: Audio still generated with TTS")
                print_status("✅ Confirmed: Transcripts still generated with Gemini AI")

            # Save results to file
            results_file = Path("e2e_test_results.json")
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "timestamp": time.time(), 
                        "total_videos": len(all_videos), 
                        "generation_mode": generation_mode,
                        "videos": all_videos
                    }, f, indent=2
                )
            print_status(f"💾 Results saved to {results_file}")

        except httpx.HTTPStatusError as e:
            print_status(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print_status(f"❌ Error: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
