from fastapi import APIRouter
from app.services.inference_service import inference_service

router = APIRouter()


@router.get("/info")
def model_info():
    return {
        "model_loaded": inference_service.model is not None,
        "labels": inference_service.labels,
        "type": type(inference_service.model).__name__ if inference_service.model else "demo-fallback",
    }