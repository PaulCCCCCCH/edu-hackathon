"""Main entry point for FastAPI application.
Only initialises the app and plugs in routers.
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

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

# Serve files generated under backend/static
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")








