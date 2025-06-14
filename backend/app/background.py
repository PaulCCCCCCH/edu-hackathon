"""Router handling user background updates."""

from fastapi import APIRouter

from .schemas import UserBackground
from .utils import state

router = APIRouter()

@router.post("/background")
async def set_background(payload: UserBackground):
    """Save background information provided from the landing-page form."""
    state.user_background = payload
    return {"status": "saved"}
