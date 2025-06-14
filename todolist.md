# TikTok-like Educational App - Todo List

## Frontend Developer (React/UI) 🎨

### Setup & Foundation
- [ ] Set up React project structure
  - Files: `package.json`, `src/App.js`, `src/index.js`
- [ ] Create onboarding flow component to collect user background and interests
  - Files: `src/components/Onboarding.js`, `src/components/TopicSelector.js`, `src/styles/onboarding.css`

### Video Feed & Interactions
- [ ] Create TikTok-like video feed component with vertical scrolling
  - Files: `src/components/VideoFeed.js`, `src/components/VideoPlayer.js`, `src/styles/video-feed.css`
- [ ] Implement double-tap like functionality for videos
  - Files: `src/components/VideoPlayer.js`, `src/hooks/useDoubleTap.js`, `src/styles/video-interactions.css`

### Data Integration & API
- [ ] Build frontend API integration for video fetching
  - Files: `src/services/api.js`, `src/hooks/useVideos.js`
- [ ] Create user preference tracking UI components
  - Files: `src/components/PreferencePanel.js`, `src/utils/tracking.js`

### Testing & Polish
- [ ] Frontend testing and component optimization
  - Files: `src/tests/`, component refinements
- [ ] Responsive design and mobile optimization
  - Files: CSS updates, responsive utilities

---

## Backend Developer (Python/AI) 🤖

### Setup & AI Integration
- [ ] Set up FastAPI backend structure
  - Files: `main.py`, `requirements.txt`
- [ ] Install and configure google-genai package for LLM integration
  - Files: `requirements.txt`, `config/ai_config.py`, `services/ai_service.py`

### Content Generation Pipeline
- [ ] Implement GPT-4o integration to generate video transcripts with metadata
  - Files: `services/content_generator.py`, `prompts/video_generation_prompts.py`, `schemas/video_schema.json`
- [ ] Integrate Google TTS API for audio generation from transcripts
  - Files: `services/audio_service.py`, `utils/audio_processing.py`
- [ ] Integrate Google Veo API for video generation from transcripts
  - Files: `services/video_service.py`, `utils/video_processing.py`
- [ ] Build video processing pipeline (transcript → audio → video → storage)
  - Files: `services/pipeline_service.py`, `workers/video_processor.py`, `utils/storage_utils.py`

### Data & Storage Management
- [ ] Build user profile data structure (JSON) to store preferences and topics
  - Files: `data/user_profile.json`, `schemas/user_schema.json`, `services/user_service.py`
- [ ] Create JSON schema for video batch (20 videos with transcript, style, topics)
  - Files: `schemas/video_batch_schema.json`, `data/video_batches/batch_001.json`
- [ ] Set up S3-compatible storage for video file storage
  - Files: `config/storage_config.py`, `services/storage_service.py`, `utils/s3_utils.py`
- [ ] Create file-based data storage system (JSON/MD files for user data)
  - Files: `data/users/user_data.json`, `data/videos/video_metadata.json`, `utils/file_storage.py`

### AI Intelligence & Recommendations
- [ ] Build preference tracking system based on user likes
  - Files: `services/preference_service.py`, `data/user_interactions.json`, `utils/analytics_utils.py`
- [ ] Create intelligent content recommendation system using LLM
  - Files: `services/recommendation_service.py`, `prompts/recommendation_prompts.py`, `utils/preference_analyzer.py`
- [ ] Implement batch video generation trigger based on user preferences
  - Files: `services/batch_generator.py`, `workers/content_scheduler.py`, `utils/trigger_utils.py`
- [ ] Add adjacent topic discovery and recommendation logic
  - Files: `services/topic_discovery.py`, `utils/topic_relations.py`, `data/topic_graph.json`

### Infrastructure & Testing
- [ ] Implement video metadata management and retrieval system
  - Files: `services/metadata_service.py`, `utils/metadata_utils.py`, `data/video_index.json`
- [ ] Add error handling and retry logic for AI API calls
  - Files: `utils/error_handler.py`, `utils/retry_logic.py`, `services/ai_service.py`
- [ ] Test end-to-end flow from onboarding to video generation and viewing
  - Files: `tests/test_e2e.py`, `tests/test_integration.py`

---

## Project Structure Overview

```
edu-hackathon/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── styles/
│   │   ├── services/
│   │   ├── utils/
│   │   └── tests/
│   └── package.json
├── backend/
│   ├── services/
│   ├── utils/
│   ├── workers/
│   ├── config/
│   ├── schemas/
│   ├── prompts/
│   ├── tests/
│   ├── main.py
│   └── requirements.txt
└── data/
    ├── users/
    ├── videos/
    └── video_batches/
```

## Key Dependencies

**Frontend:**
- React
- CSS for styling
- Custom hooks for interactions

**Backend:**
- FastAPI
- google-genai
- Pydantic (for schemas)
- File I/O libraries