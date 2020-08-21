from backend.api_v1.endpoints import model
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(model.router, prefix="/model", tags=["model"])
