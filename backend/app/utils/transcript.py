"""Utility functions for transcript generation."""

from ..schemas import UserBackground


def generate_transcript(background: UserBackground, topic: str) -> str:
    """Create a short transcript customised to the user's background."""
    return (
        f"Hi there! This 30-second tutorial on {topic} is crafted for "
        f"a {background.education_level}-level {background.job}. Enjoy learning!"
    )
