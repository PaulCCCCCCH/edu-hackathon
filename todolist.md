# TikTok-like Educational App - Todo List

## Frontend Developer (React/UI) 🎨

### Setup & Foundation

- [ ] Set up React project structure
  - Files: `package.json`, `src/App.js`, `src/index.js`
- [ ] Create onboarding flow component to collect user background and interests
  - Files: `src/components/Onboarding.js`, `src/components/TopicSelector.js`, `src/styles/onboarding.css`

### Video Feed & Interactions

- [x] Create TikTok-like video feed component with vertical scrolling
  - Files: `src/components/VideoFeed.js`, `src/components/VideoPlayer.js`, `src/styles/video-feed.css`
- [ ] Implement double-tap like functionality for videos
  - Files: `src/components/VideoPlayer.js`, `src/hooks/useDoubleTap.js`, `src/styles/video-interactions.css`

### Data Integration & API

- [ ] Build frontend API integration for video fetching
  - Files: `src/services/api.js`, `src/hooks/useVideos.js`
- [ ] Create user preference tracking UI components
  - Files: `src/components/PreferencePanel.js`, `src/utils/tracking.js`

---

## Backend Developer (Python/AI) 🤖

### Content Generation Pipeline

- [x] Implement Gemini integration to generate video transcripts with metadata
  - Files: `services/content_generator.py`, `prompts/video_generation_prompts.py`, `schemas/video_schema.json`
- [x] Integrate Google TTS API for audio generation from transcripts
  - Files: `services/audio_service.py`, `utils/audio_processing.py`
- [x] Integrate Gemini for video generation from transcripts
  - Files: `services/video_service.py`, `utils/video_processing.py`
- [x] Build video processing pipeline (transcript → audio → video → local storage)
  - Files: `services/pipeline_service.py`, `workers/video_processor.py`, `utils/storage_utils.py`

### Data & Storage Management

- [x] Build user profile data structure (JSON) to store preferences and topics
  - Files: `data/user_profile.json`, `schemas/user_schema.json`, `services/user_service.py`
- [x] Create JSON schema for video batch (20 videos with transcript, style, topics)
  - Files: `schemas/video_batch_schema.json`, `data/video_batches/batch_001.json`
- [x] Set up local storage for video files in storage/ directory
  - Files: `services/storage_service.py`, `utils/local_storage.py`, `storage/videos/`
- [x] Create file-based data storage system (JSON/MD files for user data)
  - Files: `data/users/user_data.json`, `data/videos/video_metadata.json`, `utils/file_storage.py`

### AI Intelligence & Recommendations

- [x] Build preference tracking system based on user likes
  - Files: `services/preference_service.py`, `data/user_interactions.json`, `utils/analytics_utils.py`
- [x] Create intelligent content recommendation system using LLM
  - Files: `services/recommendation_service.py`, `prompts/recommendation_prompts.py`, `utils/preference_analyzer.py`
- [x] Implement batch video generation trigger based on user preferences
  - Files: `services/batch_generator.py`, `workers/content_scheduler.py`, `utils/trigger_utils.py`
- [x] Add adjacent topic discovery and recommendation logic
  - Files: `services/topic_discovery.py`, `utils/topic_relations.py`, `data/topic_graph.json`

### Infrastructure & Testing

- [x] Implement video metadata management and retrieval system
  - Files: `services/metadata_service.py`, `utils/metadata_utils.py`, `data/video_index.json`
- [x] Add error handling and retry logic for AI API calls
  - Files: `utils/error_handler.py`, `utils/retry_logic.py`, `services/ai_service.py`

---

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
