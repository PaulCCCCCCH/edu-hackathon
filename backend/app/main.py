"""Main entry point for FastAPI application.
Only initialises the app and plugs in routers.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .background import router as background_router
from .video import router as video_router


app = FastAPI(title="EduTok Demo API")

# Allow the React dev server (default :3000) & any other origins during demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # loosen for hackathon simplicity
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(background_router)
app.include_router(video_router)

# Serve files generated under backend/static and storage
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve storage directories for videos and audio
STORAGE_DIR = BASE_DIR.parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")








