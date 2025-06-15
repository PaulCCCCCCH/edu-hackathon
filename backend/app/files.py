"""Router for serving local video and audio files."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


router = APIRouter()

# Local storage paths
STORAGE_ROOT = Path("/home/samdc/Dev/edu-hackathon/storage")
VIDEO_DIR = STORAGE_ROOT / "videos"
AUDIO_DIR = STORAGE_ROOT / "audio"

@router.get("/files/videos/{filename}")
async def serve_video(filename: str):
    """Serve video files from local storage."""
    video_path = VIDEO_DIR / filename

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Invalid video file")

    # Security check - ensure file is within video directory
    try:
        video_path.resolve().relative_to(VIDEO_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )

@router.get("/files/audio/{filename}")
async def serve_audio(filename: str):
    """Serve audio files from local storage."""
    audio_path = AUDIO_DIR / filename

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Invalid audio file")

    # Security check - ensure file is within audio directory
    try:
        audio_path.resolve().relative_to(AUDIO_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )

@router.get("/files/info")
async def get_storage_info():
    """Get information about local storage."""
    video_count = len([f for f in VIDEO_DIR.glob("*.mp4")]) if VIDEO_DIR.exists() else 0
    audio_count = len([f for f in AUDIO_DIR.glob("*.mp3")]) if AUDIO_DIR.exists() else 0

    # Calculate storage usage
    video_size = sum(f.stat().st_size for f in VIDEO_DIR.glob("*") if f.is_file()) if VIDEO_DIR.exists() else 0
    audio_size = sum(f.stat().st_size for f in AUDIO_DIR.glob("*") if f.is_file()) if AUDIO_DIR.exists() else 0

    return {
        "video_files": video_count,
        "audio_files": audio_count,
        "video_storage_bytes": video_size,
        "audio_storage_bytes": audio_size,
        "total_storage_bytes": video_size + audio_size,
        "storage_path": str(STORAGE_ROOT)
    }
