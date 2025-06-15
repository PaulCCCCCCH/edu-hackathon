"""Service for video generation using Google Gemini API."""

import asyncio
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""

    prompt: str
    duration_seconds: float = 30.0
    resolution: str = "720p"
    style: str = "educational"
    audio_file_path: str | None = None
    transcript: str | None = None


class VideoGenerationResult(BaseModel):
    """Result of video generation."""

    video_file_path: str
    thumbnail_path: str | None = None
    duration_seconds: float
    file_size_bytes: int
    aspect_ratio: str
    generation_method: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class VideoService:
    """Service for generating videos using Google Gemini API."""

    def __init__(self, storage_path: str | None = None, api_key: str | None = None) -> None:
        """Initialize the video service."""
        self.storage_path = Path(storage_path or "../storage/videos")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = "gemini-2.0-flash-exp"  # Using available model

        # Video generation settings from environment
        self.use_short_videos = os.getenv("USE_SHORT_VIDEOS", "true").lower() == "true"
        self.short_video_duration = int(os.getenv("SHORT_VIDEO_DURATION", "10"))
        self.enable_video_looping = os.getenv("ENABLE_VIDEO_LOOPING", "true").lower() == "true"
        self.video_generation_mode = os.getenv("VIDEO_GENERATION_MODE", "ai").lower()
        self.video_samples_count = int(os.getenv("VIDEO_SAMPLES_COUNT", "3"))
        self.video_samples_path = Path("../storage/video_samples")

        logger.info(f"Video settings: mode={self.video_generation_mode}, short_videos={self.use_short_videos}, "
                   f"duration={self.short_video_duration}s, looping={self.enable_video_looping}, samples_count={self.video_samples_count}")

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. Video generation will be mocked.")
            self.client = None
        else:
            try:
                # Initialize the Google Genai client
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Google Genai client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Genai client: {e}. Video generation will be mocked.")
                self.client = None

    async def generate_video_from_transcript(
        self,
        transcript: str,
        video_id: str,
        audio_file_path: str | None = None,
        style: str = "explanation",
        duration_seconds: float = 30.0,
    ) -> VideoGenerationResult:
        """Generate video from transcript and audio."""
        # Create a detailed prompt for video generation based on the transcript
        video_prompt = self._create_video_prompt(transcript, style)

        request = VideoGenerationRequest(
            prompt=video_prompt,
            duration_seconds=duration_seconds,
            style=style,
            audio_file_path=audio_file_path,
            transcript=transcript,
        )

        return await self.generate_video(request, video_id)

    async def generate_video(self, request: VideoGenerationRequest, video_id: str) -> VideoGenerationResult:
        """Generate video based on the request."""
        output_filename = f"{video_id}_video.mp4"
        output_path = self.storage_path / output_filename

        # Check video generation mode
        if self.video_generation_mode == "samples":
            logger.info("Using video samples mode")
            return await self._generate_video_from_samples(request, output_path)
        
        # Original AI generation mode
        if not self.client:
            # Fall back to mock if no client available
            return await self._mock_video_generation(request, output_path)

        try:
            # Try real video generation with Gemini Veo
            return await self._generate_video_with_gemini(request, output_path)
        except Exception as e:
            logger.exception(f"Error with real video generation: {e}")
            # Fall back to mock on any error
            logger.warning("Falling back to mock video generation")
            return await self._mock_video_generation(request, output_path)

    async def _generate_video_with_gemini(self, request: VideoGenerationRequest, output_path: Path) -> VideoGenerationResult:
        """Generate video using Gemini Veo API."""
        try:
            # Determine the video duration to generate
            if self.use_short_videos:
                generation_duration = self.short_video_duration
                logger.info(f"Generating short video: {generation_duration}s (will loop to {request.duration_seconds}s if needed)")
            else:
                generation_duration = int(request.duration_seconds)
                logger.info(f"Generating full-length video: {generation_duration}s")

            # Create an optimized prompt for the actual generation duration
            optimized_prompt = self._create_duration_optimized_prompt(request.prompt, generation_duration)

            logger.info(f"Starting Gemini Veo video generation for: {optimized_prompt[:100]}...")

            # Create video generation operation
            operation = self.client.models.generate_videos(
                model="veo-2.0-generate-001",
                prompt=optimized_prompt,
                config=types.GenerateVideosConfig(
                    person_generation="dont_allow",
                    aspect_ratio="16:9",
                )
            )

            logger.info(f"Video generation started (Operation: {operation.name})")

            # Poll for completion - video generation can take several minutes
            poll_count = 0
            max_polls = 180  # 60 minutes max (20 seconds * 180)

            while not operation.done:
                poll_count += 1

                if poll_count >= max_polls:
                    logger.error("Video generation timeout - falling back to mock")
                    return await self._mock_video_generation(request, output_path)

                # Log progress every 5 polls (every ~2 minutes)
                if poll_count % 5 == 0:
                    logger.info(f"Video generation in progress... (poll {poll_count}/{max_polls})")

                # Wait 20 seconds between polls
                await asyncio.sleep(20)

                try:
                    operation = self.client.operations.get(operation)
                except Exception as poll_error:
                    logger.warning(f"Error polling operation: {poll_error}")
                    # Continue trying for a few more attempts
                    if poll_count > 10:  # Give up after 10 failed polls
                        logger.error("Too many polling failures - falling back to mock")
                        return await self._mock_video_generation(request, output_path)
                    continue

            logger.info("Video generation completed!")

            # Check if we have result data
            if not hasattr(operation, "result") or not operation.result:
                logger.error("No result data in completed operation")
                return await self._mock_video_generation(request, output_path)

            # Download the generated video
            if hasattr(operation.result, "generated_videos") and operation.result.generated_videos:
                generated_video = operation.result.generated_videos[0]  # Take first video
                video = generated_video.video

                # Download video data
                try:
                    if hasattr(video, "data"):
                        video_data = video.data
                    elif hasattr(video, "download"):
                        video_data = video.download()
                    else:
                        # Try to use the file download method
                        video_data = self.client.files.download(file=video)

                    # Save the original generated video
                    temp_path = output_path.with_suffix(".temp.mp4")
                    with open(temp_path, "wb") as f:
                        f.write(video_data)

                    original_size = temp_path.stat().st_size
                    logger.info(f"Original video saved: {temp_path} ({original_size:,} bytes)")

                    # Process the video (loop if needed and combine with audio)
                    final_video_path = await self._process_generated_video(
                        temp_path, output_path, generation_duration, request.duration_seconds, request.audio_file_path
                    )

                    final_size = final_video_path.stat().st_size
                    actual_duration = request.duration_seconds

                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()

                    return VideoGenerationResult(
                        video_file_path=str(final_video_path),
                        thumbnail_path=None,
                        duration_seconds=actual_duration,
                        file_size_bytes=final_size,
                        aspect_ratio="16:9",
                        generation_method="gemini_veo" + ("_looped" if self.use_short_videos and self.enable_video_looping else ""),
                        metadata={
                            "prompt": request.prompt,
                            "style": request.style,
                            "transcript_length": len(request.transcript) if request.transcript else 0,
                            "has_audio": request.audio_file_path is not None,
                            "operation_name": operation.name,
                            "model_used": "veo-2.0-generate-001",
                            "original_duration": generation_duration,
                            "final_duration": actual_duration,
                            "was_looped": self.use_short_videos and self.enable_video_looping and generation_duration < request.duration_seconds
                        },
                    )

                except Exception as download_error:
                    logger.exception(f"Error downloading video: {download_error}")
                    return await self._mock_video_generation(request, output_path)
            else:
                logger.error("No video data found in result")
                return await self._mock_video_generation(request, output_path)

        except Exception as e:
            logger.exception(f"Error generating video with Gemini Veo: {e}")

            # Check for specific error types
            if "fps parameter is not supported" in str(e):
                logger.error("Veo API parameter error - check configuration")
            elif "model 'veo-2.0-generate-001' does not exist" in str(e):
                logger.error("Veo model not available - requires allowlist access")

            # Fall back to mock generation
            return await self._mock_video_generation(request, output_path)

    async def _generate_video_from_samples(self, request: VideoGenerationRequest, output_path: Path) -> VideoGenerationResult:
        """Generate video by stitching together random samples from video_samples directory."""
        try:
            # Get all video files from samples directory
            sample_files = list(self.video_samples_path.glob("*.mp4"))
            
            if not sample_files:
                logger.warning("No video samples found, falling back to mock generation")
                return await self._mock_video_generation(request, output_path)
            
            # Randomly select videos to stitch
            num_samples = min(self.video_samples_count, len(sample_files))
            selected_samples = random.sample(sample_files, num_samples)
            
            logger.info(f"Selected {num_samples} video samples to stitch: {[f.name for f in selected_samples]}")
            
            # Get target duration (from audio if available, otherwise use requested duration)
            target_duration = request.duration_seconds
            if request.audio_file_path and Path(request.audio_file_path).exists():
                try:
                    import subprocess
                    audio_duration_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(request.audio_file_path)]
                    audio_result = subprocess.run(audio_duration_cmd, capture_output=True, text=True, timeout=30, check=False)
                    if audio_result.returncode == 0:
                        target_duration = float(audio_result.stdout.strip())
                        logger.info(f"Using audio duration as target: {target_duration:.2f}s")
                except Exception as e:
                    logger.warning(f"Could not get audio duration: {e}")
            
            # Stitch videos together with target duration (this will create 5s segments and loop to fill duration)
            stitched_video_path = await self._stitch_video_samples(selected_samples, output_path, target_duration)
            
            # If audio is provided, combine with the stitched video
            if request.audio_file_path and Path(request.audio_file_path).exists():
                final_video_path = await self._combine_video_audio(
                    stitched_video_path, 
                    output_path.with_suffix(".final.mp4"), 
                    request.audio_file_path
                )
                # Replace the output with the final version
                if final_video_path != output_path:
                    final_video_path.rename(output_path)
                actual_duration = target_duration
            else:
                actual_duration = target_duration
            
            file_size = output_path.stat().st_size
            
            logger.info(f"Video generated from samples: {output_path} (duration: {actual_duration:.2f}s)")
            
            return VideoGenerationResult(
                video_file_path=str(output_path),
                thumbnail_path=None,
                duration_seconds=actual_duration,
                file_size_bytes=file_size,
                aspect_ratio="16:9",
                generation_method="video_samples",
                metadata={
                    "prompt": request.prompt,
                    "style": request.style,
                    "transcript_length": len(request.transcript) if request.transcript else 0,
                    "has_audio": request.audio_file_path is not None,
                    "samples_used": [f.name for f in selected_samples],
                    "samples_count": num_samples,
                },
            )
            
        except Exception as e:
            logger.exception(f"Error generating video from samples: {e}")
            # Fall back to mock generation
            return await self._mock_video_generation(request, output_path)

    async def _stitch_video_samples(self, sample_files: list[Path], output_path: Path, target_duration: float = None) -> Path:
        """Stitch multiple video samples together using 5-second segments from each, looping to fill target duration."""
        try:
            import subprocess
            import tempfile
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            segment_duration = 5.0  # Use 5 seconds from each sample
            logger.info(f"Creating video from {len(sample_files)} samples using {segment_duration}s segments")
            
            if target_duration:
                logger.info(f"Target duration: {target_duration:.2f}s")
            
            # OPTIMIZATION: Try direct concatenation with trimming first (fastest method)
            if len(sample_files) == 1 and target_duration:
                logger.info("Single sample file - using direct loop/trim method for maximum speed")
                return await self._direct_loop_video(sample_files[0], output_path, segment_duration, target_duration)
            
            # Create 5-second segments from each sample first
            segment_files = []
            temp_dir = Path(tempfile.mkdtemp())
            
            try:
                # Step 1: Extract 5-second segments from each sample (PARALLEL PROCESSING)
                async def create_segment(i: int, sample_file: Path) -> tuple[int, Path | None]:
                    """Create a single segment asynchronously"""
                    segment_file = temp_dir / f"segment_{i:03d}.mp4"
                    
                    # Use hardware acceleration if available, with fast presets
                    cmd_segment = [
                        "ffmpeg", "-y",
                        "-hwaccel", "auto",  # Auto-detect hardware acceleration
                        "-i", str(sample_file),
                        "-t", str(segment_duration),  # Take only first 5 seconds
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-preset", "ultrafast",  # Faster preset
                        "-crf", "28",  # Slightly lower quality for speed
                        str(segment_file)
                    ]
                    
                    def run_ffmpeg():
                        return subprocess.run(cmd_segment, capture_output=True, text=True, timeout=30, check=False)
                    
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(executor, run_ffmpeg)
                    
                    if result.returncode == 0:
                        logger.info(f"Created {segment_duration}s segment from {sample_file.name}")
                        return i, segment_file
                    else:
                        logger.warning(f"Failed to create segment from {sample_file.name}: {result.stderr}")
                        return i, None
                
                # Run all segment creation tasks concurrently
                tasks = [create_segment(i, sample_file) for i, sample_file in enumerate(sample_files)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect successful segments in order
                for i, segment_file in results:
                    if isinstance(segment_file, Exception):
                        logger.error(f"Segment creation failed with exception: {segment_file}")
                    elif segment_file is not None:
                        segment_files.append(segment_file)
                
                if not segment_files:
                    raise Exception("No segments could be created from samples")
                
                # Step 2: Calculate how many loops we need to fill target duration
                if target_duration:
                    total_segment_duration = len(segment_files) * segment_duration
                    loops_needed = max(1, int(target_duration / total_segment_duration) + 1)
                    logger.info(f"Need {loops_needed} loops of {total_segment_duration:.1f}s segments to fill {target_duration:.2f}s")
                else:
                    loops_needed = 1
                    logger.info("No target duration specified, using segments once")
                
                # Step 3: Create concat file with repeated segments
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    concat_file = f.name
                    for loop in range(loops_needed):
                        for segment_file in segment_files:
                            f.write(f"file '{segment_file.absolute()}'\n")
                
                # Step 4: Concatenate all segments with hardware acceleration
                cmd_concat = [
                    "ffmpeg", "-y",
                    "-hwaccel", "auto",  # Hardware acceleration for decoding
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_file,
                    "-c", "copy",  # Copy streams without re-encoding for speed
                    "-avoid_negative_ts", "make_zero",  # Fix timestamp issues
                    str(output_path)
                ]
                
                if target_duration:
                    # Trim to exact target duration
                    cmd_concat.extend(["-t", str(target_duration)])
                
                result = subprocess.run(cmd_concat, capture_output=True, text=True, timeout=60, check=False)  # Reduced timeout
                
                if result.returncode == 0:
                    logger.info(f"Successfully created video with {segment_duration}s segments: {output_path}")
                    return output_path
                else:
                    logger.warning(f"Concat failed, trying re-encoding method: {result.stderr}")
                    
                    # Fallback: re-encode during concat with hardware acceleration
                    cmd_reencode = [
                        "ffmpeg", "-y",
                        "-hwaccel", "auto",
                        "-f", "concat",
                        "-safe", "0", 
                        "-i", concat_file,
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-preset", "ultrafast",  # Faster preset
                        "-crf", "28",  # Faster encoding
                        "-avoid_negative_ts", "make_zero",
                        str(output_path)
                    ]
                    
                    if target_duration:
                        cmd_reencode.extend(["-t", str(target_duration)])
                    
                    result2 = subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=120, check=False)  # Reduced timeout
                    
                    if result2.returncode == 0:
                        logger.info(f"Successfully created video with re-encoding: {output_path}")
                        return output_path
                    else:
                        logger.error(f"Re-encoding also failed: {result2.stderr}")
                        raise Exception(f"ffmpeg concat failed: {result2.stderr}")
                        
            finally:
                # Clean up temp files and directory
                try:
                    Path(concat_file).unlink()
                except:
                    pass
                    
                for segment_file in segment_files:
                    try:
                        segment_file.unlink()
                    except:
                        pass
                        
                try:
                    temp_dir.rmdir()
                except:
                    pass
                    
        except Exception as e:
            logger.exception(f"Error stitching video samples: {e}")
            # Fallback: just use the first 5 seconds of the first sample
            if sample_files:
                logger.info("Falling back to using first 5 seconds of first sample only")
                try:
                    import subprocess
                    cmd_fallback = [
                        "ffmpeg", "-y",
                        "-i", str(sample_files[0]),
                        "-t", "5.0",  # Just 5 seconds
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        str(output_path)
                    ]
                    subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=60, check=True)
                    return output_path
                except:
                    # Last resort: copy the whole first file
                    import shutil
                    shutil.copy2(sample_files[0], output_path)
                    return output_path
            else:
                raise e
    
    async def _direct_loop_video(self, sample_file: Path, output_path: Path, segment_duration: float, target_duration: float) -> Path:
        """Directly loop a single video file for maximum speed - avoids creating intermediate segments."""
        try:
            import subprocess
            
            logger.info(f"Using direct loop method for {sample_file} -> {output_path}")
            
            # Calculate how many loops we need
            loops_needed = max(1, int(target_duration / segment_duration) + 1)
            logger.info(f"Looping {segment_duration}s segment {loops_needed} times to fill {target_duration:.2f}s")
            
            # Single FFmpeg command that does everything at once
            cmd = [
                "ffmpeg", "-y",
                "-hwaccel", "auto",
                "-stream_loop", str(loops_needed - 1),  # Loop the input
                "-i", str(sample_file),
                "-t", str(target_duration),  # Trim to exact duration
                "-ss", "0",  # Start from beginning
                "-c:v", "libx264",
                "-c:a", "aac", 
                "-preset", "ultrafast",
                "-crf", "28",
                "-avoid_negative_ts", "make_zero",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
            
            if result.returncode == 0:
                logger.info(f"Direct loop method successful: {output_path}")
                return output_path
            else:
                logger.warning(f"Direct loop failed: {result.stderr}")
                # Fall back to segment method
                raise Exception("Direct loop failed, falling back to segment method")
                
        except Exception as e:
            logger.warning(f"Direct loop method failed: {e}")
            # This will cause the caller to fall back to the segment-based method
            raise

    async def _mock_video_generation(self, request: VideoGenerationRequest, output_path: Path) -> VideoGenerationResult:
        """Mock video generation for testing and development."""
        # Create a simple mock video file with proper duration
        mock_video_content = self._create_mock_video_content(request)

        # If audio is provided, combine with mock video and ensure proper duration
        if request.audio_file_path and Path(request.audio_file_path).exists():
            temp_path = output_path.with_suffix(".temp.mp4")
            with open(temp_path, "wb") as f:
                f.write(mock_video_content)

            # Try to combine with audio (this will handle looping if needed to match full audio duration)
            final_path = await self._combine_video_audio(temp_path, output_path, request.audio_file_path)

            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

            # Get the actual duration from the audio file for the result - this is the final video duration
            try:
                import subprocess
                audio_duration_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(request.audio_file_path)]
                audio_result = subprocess.run(audio_duration_cmd, capture_output=True, text=True, timeout=30, check=False)
                if audio_result.returncode == 0:
                    actual_duration = float(audio_result.stdout.strip())
                    logger.info(f"Mock video will match audio duration: {actual_duration:.2f}s")
                else:
                    actual_duration = request.duration_seconds
                    logger.warning("Could not get audio duration, using requested duration")
            except Exception as e:
                logger.warning(f"Error getting audio duration: {e}")
                actual_duration = request.duration_seconds
        else:
            with open(output_path, "wb") as f:
                f.write(mock_video_content)
            actual_duration = request.duration_seconds

        file_size = output_path.stat().st_size

        logger.info(f"Mock video generated: {output_path} (duration: {actual_duration:.2f}s)")

        return VideoGenerationResult(
            video_file_path=str(output_path),
            thumbnail_path=None,
            duration_seconds=actual_duration,
            file_size_bytes=file_size,
            aspect_ratio="16:9",
            generation_method="mock",
            metadata={
                "prompt": request.prompt,
                "style": request.style,
                "transcript_length": len(request.transcript) if request.transcript else 0,
                "has_audio": request.audio_file_path is not None,
                "looped_to_audio_duration": request.audio_file_path is not None,
            },
        )

    def _create_video_prompt(self, transcript: str, style: str) -> str:
        """Create a detailed video generation prompt based on transcript and style."""
        # Extract key concepts from transcript for visual generation
        key_concepts = self._extract_visual_concepts(transcript)

        style_guidelines = {
            "explanation": "Clear, professional educational video with simple graphics and text overlays",
            "tutorial": "Step-by-step tutorial format with hands-on demonstrations and clear visual cues",
            "story": "Narrative-driven video with engaging visuals and smooth transitions",
            "quiz": "Interactive quiz format with questions, options, and visual feedback",
        }

        return f"""Create an educational video in {style} style.

Content: {transcript[:500]}{"..." if len(transcript) > 500 else ""}

Visual elements to include: {", ".join(key_concepts)}

Style guidelines: {style_guidelines.get(style, "Educational video format")}

Requirements:
- Duration: approximately {30} seconds
- Clear, engaging visuals that support the educational content
- Smooth transitions between scenes
- Text overlays for key points
- Educational and professional tone
"""

    def _extract_visual_concepts(self, transcript: str) -> list[str]:
        """Extract visual concepts from transcript for video generation."""
        # Simple keyword extraction for visual elements
        visual_keywords = [
            "python",
            "code",
            "variable",
            "function",
            "data",
            "algorithm",
            "graph",
            "chart",
            "diagram",
            "example",
            "step",
            "process",
            "concept",
            "idea",
            "theory",
            "practice",
            "tutorial",
            "lesson",
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

    def _create_duration_optimized_prompt(self, original_prompt: str, target_duration: int) -> str:
        """Create a prompt optimized for a specific duration."""
        if target_duration <= 10:
            return f"Create a {target_duration}-second loopable educational animation: {original_prompt}. The video should be designed to seamlessly loop and repeat."
        return f"Create a {target_duration}-second educational video: {original_prompt}"

    async def _process_generated_video(self, input_path: Path, output_path: Path, original_duration: int, target_duration: float, audio_file_path: str | None = None) -> Path:
        """Process the generated video (loop if needed to match audio duration)."""
        try:
            # Always use _combine_video_audio if audio is available - it handles all looping
            if audio_file_path and Path(audio_file_path).exists():
                logger.info(f"Combining {original_duration}s video with audio to match full audio duration")
                return await self._combine_video_audio(input_path, output_path, audio_file_path)
            
            # If no audio and we don't need to loop, just rename
            if not self.use_short_videos or not self.enable_video_looping or original_duration >= target_duration:
                input_path.rename(output_path)
                logger.info(f"Video processed without looping: {output_path}")
                return output_path

            # Loop video only (without external audio)
            loops_needed = int(target_duration / original_duration) + 1
            logger.info(f"Looping {original_duration}s video {loops_needed} times to reach ~{target_duration}s (no external audio)")

            try:
                import subprocess

                # Create a filter that loops the video
                filter_complex = f"[0:v]loop=loop={loops_needed-1}:size=1:start=0[outv];[0:a]aloop=loop={loops_needed-1}:size=1:start=0[outa]"

                cmd = [
                    "ffmpeg", "-y", "-i", str(input_path),
                    "-filter_complex", filter_complex,
                    "-map", "[outv]", "-map", "[outa]",
                    "-t", str(target_duration),  # Trim to exact duration
                    "-c:v", "libx264", "-c:a", "aac",
                    str(output_path)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

                if result.returncode == 0:
                    logger.info(f"Successfully looped video using ffmpeg: {output_path}")
                    return output_path
                logger.warning(f"ffmpeg failed: {result.stderr}")
                # Fall back to simple copy
                input_path.rename(output_path)
                return output_path

            except (subprocess.TimeoutExpired, FileNotFoundError, ImportError) as e:
                logger.warning(f"ffmpeg not available or failed: {e}")
                # Fall back to simple copy
                input_path.rename(output_path)
                return output_path

        except Exception as e:
            logger.exception(f"Error processing video: {e}")
            # Fall back to simple copy
            try:
                input_path.rename(output_path)
            except:
                # If rename fails, copy the content
                with open(input_path, "rb") as src, open(output_path, "wb") as dst:
                    dst.write(src.read())
            return output_path

    def _create_mock_video_content(self, request: VideoGenerationRequest) -> bytes:
        """Create mock video file using ffmpeg to ensure it's properly formatted."""
        try:
            import subprocess
            import tempfile
            
            # Always use short video duration for mock generation when looping is enabled
            # This ensures the mock video can be properly looped to match audio duration
            if self.use_short_videos and self.enable_video_looping:
                mock_duration = self.short_video_duration
                logger.info(f"Creating {mock_duration}s mock video (will be looped to match audio duration)")
            else:
                mock_duration = int(request.duration_seconds)
                logger.info(f"Creating {mock_duration}s mock video (no looping)")

            # Create a proper video file using ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Generate a simple test pattern video with ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=blue:s=1280x720:d={mock_duration}",  # Blue color video
                "-f", "lavfi", 
                "-i", f"sine=frequency=440:sample_rate=44100:duration={mock_duration}",  # Simple sine wave audio
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",  # Ensure compatibility
                "-preset", "ultrafast",  # Fast encoding
                "-c:a", "aac",
                "-shortest",  # End when shortest stream ends
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            
            if result.returncode == 0:
                # Read the generated video file
                with open(temp_path, "rb") as f:
                    video_content = f.read()
                
                # Clean up temp file
                try:
                    Path(temp_path).unlink()
                except:
                    pass
                    
                logger.info(f"Created proper mock video: {len(video_content)} bytes for {mock_duration}s")
                return video_content
            else:
                logger.warning(f"Failed to create mock video with ffmpeg: {result.stderr}")
                # Fall back to simple mock
                
        except Exception as e:
            logger.warning(f"Error creating mock video with ffmpeg: {e}")
        
        # Fallback: create a minimal MP4 structure
        # This is a last resort if ffmpeg is not available
        ftyp_box = (
            b"\x00\x00\x00\x20"  # Box size (32 bytes)
            b"ftyp"              # Box type
            b"isom"              # Major brand
            b"\x00\x00\x02\x00"  # Minor version
            b"isomiso2mp41"      # Compatible brands
        )
        
        # Create a more complete MP4 structure with moov atom
        moov_data = b"moov" + b"\x00" * 100  # Minimal moov box
        mdat_data = b"mdat" + b"\x00" * 1000  # Minimal mdat box
        
        mock_content = ftyp_box + moov_data + mdat_data
        logger.debug(f"Created fallback mock video content: {len(mock_content)} bytes")
        return mock_content

    # Cache for duration information to avoid repeated FFprobe calls
    _duration_cache = {}
    
    async def _get_duration(self, file_path: str) -> float | None:
        """Get duration of media file with caching"""
        if file_path in self._duration_cache:
            return self._duration_cache[file_path]
        
        import subprocess
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            self._duration_cache[file_path] = duration
            return duration
        return None
    
    async def _combine_video_audio(self, video_path: Path, output_path: Path, audio_path: str) -> Path:
        """Combine video with audio using ffmpeg, extending video to match audio duration."""
        try:
            import subprocess

            logger.info(f"Combining video {video_path} with audio {audio_path}")

            # Get durations using cached method
            audio_duration = await self._get_duration(str(audio_path))
            video_duration = await self._get_duration(str(video_path))
            
            if audio_duration:
                logger.info(f"Audio duration: {audio_duration:.2f}s")
            if video_duration:
                logger.info(f"Video duration: {video_duration:.2f}s")

            # Always loop the video to match audio duration if audio is longer
            # This ensures the final video matches the full audio length
            if audio_duration and video_duration and audio_duration > video_duration:
                loops_needed = int(audio_duration / video_duration) + 1
                logger.info(f"Audio ({audio_duration:.2f}s) longer than video ({video_duration:.2f}s), looping video {loops_needed} times to match full audio duration")

                cmd = [
                    "ffmpeg", "-y",
                    "-hwaccel", "auto",      # Hardware acceleration
                    "-stream_loop", str(loops_needed - 1),  # Loop the input stream
                    "-i", str(video_path),   # Input video (will be looped)
                    "-i", str(audio_path),   # Input audio
                    "-map", "0:v:0",         # Map video stream
                    "-map", "1:a:0",         # Map audio stream
                    "-t", str(audio_duration),  # Trim to exact audio duration
                    "-c:v", "libx264",       # Re-encode video to ensure smooth looping
                    "-preset", "ultrafast",  # Faster encoding
                    "-crf", "28",            # Faster quality setting
                    "-c:a", "copy",          # Copy audio as-is (no re-encoding needed)
                    "-avoid_negative_ts", "make_zero",  # Handle timestamp issues
                    str(output_path)
                ]
            elif audio_duration and video_duration and audio_duration < video_duration:
                # Audio is shorter than video - trim video to audio length
                logger.info(f"Video ({video_duration:.2f}s) longer than audio ({audio_duration:.2f}s), trimming video to audio length")
                cmd = [
                    "ffmpeg", "-y",
                    "-hwaccel", "auto",      # Hardware acceleration
                    "-i", str(video_path),  # Input video
                    "-i", str(audio_path),  # Input audio
                    "-map", "0:v:0",        # Map video from first input
                    "-map", "1:a:0",        # Map audio from second input
                    "-t", str(audio_duration), # Trim to exact audio duration
                    "-c:v", "libx264",      # Re-encode video
                    "-preset", "ultrafast",  # Faster encoding
                    "-crf", "28",            # Faster quality
                    "-c:a", "copy",         # Copy audio as-is
                    str(output_path)
                ]
            else:
                # Durations match or we couldn't get duration info - simple combination
                logger.info("Video and audio durations match or unknown, using simple combination")
                cmd = [
                    "ffmpeg", "-y",
                    "-hwaccel", "auto",      # Hardware acceleration
                    "-i", str(video_path),  # Input video
                    "-i", str(audio_path),  # Input audio
                    "-c:v", "copy",         # Copy video without re-encoding
                    "-c:a", "aac",          # Encode audio as AAC
                    "-map", "0:v:0",        # Map video from first input
                    "-map", "1:a:0",        # Map audio from second input
                    "-shortest",            # End when shortest stream ends
                    str(output_path)
                ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)  # Reduced timeout

            if result.returncode == 0:
                logger.info(f"Successfully combined video and audio: {output_path}")
                return output_path
            logger.warning(f"ffmpeg failed to combine video/audio: {result.stderr}")
            # Fall back to video only
            video_path.rename(output_path)
            return output_path

        except Exception as e:
            logger.warning(f"Error combining video and audio: {e}")
            # Fall back to video only
            video_path.rename(output_path)
            return output_path

    async def _loop_video_with_audio(self, video_path: Path, output_path: Path, audio_path: str, loops_needed: int, target_duration: float) -> Path:
        """Loop video and combine with audio using ffmpeg."""
        try:
            import subprocess

            logger.info(f"Looping video {loops_needed} times and combining with audio {audio_path} for {target_duration}s")

            # Create a proper looping filter that repeats the video for the full duration
            # Use concat filter to repeat the video multiple times, then trim to exact audio duration
            inputs = " ".join(["[0:v]" for _ in range(loops_needed)])
            filter_complex = f"{inputs}concat=n={loops_needed}:v=1:a=0[looped_video]"

            cmd = [
                "ffmpeg", "-y",
                "-stream_loop", str(loops_needed - 1),  # Loop the input video
                "-i", str(video_path),   # Input video (looped)
                "-i", str(audio_path),   # Input audio
                "-filter_complex", filter_complex,
                "-map", "[looped_video]", # Map looped video
                "-map", "1:a:0",         # Map audio from second input
                "-t", str(target_duration), # Trim to exact audio duration
                "-c:v", "libx264",       # Re-encode video
                "-c:a", "aac",           # Encode audio as AAC
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

            if result.returncode == 0:
                logger.info(f"Successfully looped video and combined with audio: {output_path}")
                return output_path
            logger.warning(f"ffmpeg failed to loop video with audio: {result.stderr}")
            logger.info("Trying alternative looping method...")

            # Try simpler approach: just repeat input and trim
            cmd_simple = [
                "ffmpeg", "-y",
                "-stream_loop", str(loops_needed - 1),  # Loop the input
                "-i", str(video_path),   # Input video (looped)
                "-i", str(audio_path),   # Input audio
                "-map", "0:v:0",         # Map video
                "-map", "1:a:0",         # Map audio
                "-t", str(target_duration), # Trim to exact audio duration
                "-c:v", "libx264",       # Re-encode video
                "-c:a", "copy",          # Copy audio as-is
                str(output_path)
            ]

            result2 = subprocess.run(cmd_simple, capture_output=True, text=True, timeout=300, check=False)

            if result2.returncode == 0:
                logger.info(f"Successfully looped video with simple method: {output_path}")
                return output_path
            logger.warning(f"Simple method also failed: {result2.stderr}")
            # Fall back to video copy
            video_path.rename(output_path)
            return output_path

        except Exception as e:
            logger.warning(f"Error looping video with audio: {e}")
            # Fall back to simple video copy
            video_path.rename(output_path)
            return output_path

    async def batch_generate_videos(self, video_requests: list[dict[str, Any]]) -> list[VideoGenerationResult]:
        """Generate multiple videos in batch."""
        results = []

        for i, req_data in enumerate(video_requests):
            try:
                request = VideoGenerationRequest(**req_data)
                video_id = req_data.get("video_id", f"batch_video_{i}")

                result = await self.generate_video(request, video_id)
                results.append(result)

                logger.info(f"Generated video {i + 1}/{len(video_requests)}: {video_id}")

            except Exception as e:
                logger.exception(f"Error generating video {i}: {e}")
                continue

        logger.info(f"Batch generation completed: {len(results)}/{len(video_requests)} videos generated")
        return results

    def get_video_info(self, video_file_path: str) -> dict[str, Any]:
        """Get information about a video file."""
        path = Path(video_file_path)

        if not path.exists():
            msg = f"Video file not found: {video_file_path}"
            raise FileNotFoundError(msg)

        return {
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "exists": True,
            "created_at": path.stat().st_ctime,
            "extension": path.suffix.lower(),
        }

    def list_generated_videos(self) -> list[dict[str, Any]]:
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
