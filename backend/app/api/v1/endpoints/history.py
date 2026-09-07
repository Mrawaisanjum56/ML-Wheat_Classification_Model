from fastapi import APIRouter, Query
from app.schemas.predict import HistoryResponse
from app.services.history_store import history_store

router = APIRouter()


@router.get("", response_model=HistoryResponse)
def get_history(limit: int = Query(20, ge=1, le=100)):
    items = history_store.list(limit=limit)
    return HistoryResponse(total=len(items), items=items)


@router.delete("")
def clear_history():
    history_store.clear()
    return {"message": "History cleared"}