"""Batch video processor for sequential generation with immediate availability."""

import asyncio

from .utils import state, transcript as transcript_utils, video as video_utils


class BatchVideoProcessor:
    def __init__(self):
        self.processing_queue: list[dict] = []
        self.is_processing = False

    async def add_video_to_queue(self, topics: list[str]) -> str:
        """Add video to processing queue and start processing if not already running."""
        if not state.user_background:
            raise ValueError("User background not set")

        primary_topic = topics[0] if topics else "your chosen topic"
        transcript = transcript_utils.generate_transcript(state.user_background, primary_topic)

        # Start concurrent generation immediately
        video_id = await video_utils.start_concurrent_generation(transcript)

        # Add to queue for tracking
        video_info = {
            "video_id": video_id,
            "topics": topics,
            "transcript": transcript,
            "started": True
        }
        self.processing_queue.append(video_info)

        # Start processing if not already running
        if not self.is_processing:
            asyncio.create_task(self._process_queue())

        return video_id

    async def _process_queue(self):
        """Process video queue one at a time."""
        self.is_processing = True

        while self.processing_queue:
            current_video = self.processing_queue[0]
            video_id = current_video["video_id"]

            # Wait for current video to complete
            while True:
                status = video_utils.get_video_status(video_id)
                if not status:
                    break

                if status.get("video_ready", False):
                    # Video is ready, remove from queue
                    self.processing_queue.pop(0)
                    break

                # Check every 0.5 seconds
                await asyncio.sleep(0.5)

            # Small delay before next video
            await asyncio.sleep(0.1)

        self.is_processing = False

    def get_queue_status(self) -> dict:
        """Get current queue status."""
        return {
            "queue_length": len(self.processing_queue),
            "is_processing": self.is_processing,
            "videos_in_queue": [
                {
                    "video_id": video["video_id"],
                    "topics": video["topics"],
                    "position": i
                }
                for i, video in enumerate(self.processing_queue)
            ]
        }

# Global batch processor instance
batch_processor = BatchVideoProcessor()
