# EduTok Demo

Minimal full-stack prototype reminiscent of TikTok that serves short educational videos generated on-demand.

## Stack

* **Backend** – Python FastAPI (no database; in-memory only)
* **Frontend** – Single `index.html` (React via CDN)

## Getting started

1. Install backend deps (ideally in a venv):
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Run API:
   ```bash
   uvicorn app.main:app --reload --port 8000 --host 0.0.0.0
   ```
3. Open `frontend/index.html` in your browser (or serve via any static file server).

That’s it! Use *Input My Background* to submit background info, then *Start Exploring* to generate/scroll educational clips.

## Setting Up API Key

Create a `.env` file with the following content under `/backend` folder. The content of the file should be

```
	GMI_API_KEY=<Copy API key from Google Doc>
```

In your code, add 		

```
from dotenv import load_dotenv
load_dotenv()  # take environment variables
```

### Notes

* Video generation is mocked (returns a public-domain MP4). Replace with your own model/service call in `backend/app/main.py`.
* Like/dislike UI exists but is non-functional backend-wise per spec.
* Everything lives in memory; restart clears data.
