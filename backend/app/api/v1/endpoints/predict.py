from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from app.core.config import settings
from app.schemas.predict import PredictResponse
from app.services.image_processing import read_image_bytes_to_pil, preprocess_for_model
from app.services.inference_service import inference_service
from app.services.history_store import history_store

router = APIRouter()


def _validate_file(file: UploadFile, content: bytes):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed size is {settings.MAX_FILE_SIZE_MB} MB.",
        )


@router.post("", response_model=PredictResponse)
async def predict_single(file: UploadFile = File(...)):
    content = await file.read()
    _validate_file(file, content)

    try:
        image = read_image_bytes_to_pil(content)
        x = preprocess_for_model(image)
        pred_class, conf, probs = inference_service.predict(x)

        history_store.add(
            filename=file.filename or "unknown",
            predicted_class=pred_class,
            confidence=conf,
            probabilities=probs,
        )

        return PredictResponse(
            predicted_class=pred_class,
            confidence=conf,
            probabilities=probs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Batch limit is 10 images per request.")

    results = []
    for file in files:
        content = await file.read()
        _validate_file(file, content)

        image = read_image_bytes_to_pil(content)
        x = preprocess_for_model(image)
        pred_class, conf, probs = inference_service.predict(x)

        item = history_store.add(
            filename=file.filename or "unknown",
            predicted_class=pred_class,
            confidence=conf,
            probabilities=probs,
        )

        results.append(item)

    return {"total": len(results), "items": results}