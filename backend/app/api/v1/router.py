from fastapi import APIRouter
from app.api.v1.endpoints import health, predict, history, model

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(model.router, prefix="/model", tags=["model"])
api_router.include_router(predict.router, prefix="/predict", tags=["predict"])
api_router.include_router(history.router, prefix="/history", tags=["history"])