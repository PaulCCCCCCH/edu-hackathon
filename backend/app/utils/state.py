"""Simple in-memory state shared across routers."""

from typing import Optional

from ..schemas import UserBackground

# This will reset whenever the server restarts
user_background: Optional[UserBackground] = None
