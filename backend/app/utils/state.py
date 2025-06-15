"""Simple in-memory state shared across routers."""


from ..schemas import UserBackground


# This will reset whenever the server restarts
user_background: UserBackground | None = None
