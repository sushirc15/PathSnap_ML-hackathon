"""FastAPI route definitions for PathSnap."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from src.api.services import PathSnapInferenceService

router = APIRouter(prefix="/api", tags=["PathSnap"])

_service: PathSnapInferenceService | None = None


def get_service() -> PathSnapInferenceService:
    global _service
    if _service is None:
        _service = PathSnapInferenceService()
    return _service


def set_service(service: PathSnapInferenceService) -> None:
    """Allow dependency injection for tests."""
    global _service
    _service = service


@router.get("/health")
def health() -> dict:
    """Return service status, loaded ML/HMM model metadata, and available trip count."""
    return get_service().get_health_info()


@router.get("/trips")
def list_trips() -> dict:
    """Return available trips from the configured evaluation directory."""
    return get_service().list_available_trips()


@router.post("/trips/{trip_id}/predict")
def predict_trip(trip_id: str) -> dict:
    """Run full map-matching pipeline (candidates -> features -> classifier -> HMM/Viterbi)."""
    try:
        return get_service().predict_and_evaluate(trip_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@router.get("/trips/{trip_id}/map", response_class=HTMLResponse)
def get_trip_map(trip_id: str) -> HTMLResponse:
    """Return evaluator-safe interactive Folium HTML map (never exposes ground truth)."""
    try:
        html = get_service().get_map_html(trip_id)
        return HTMLResponse(content=html, status_code=200)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Map generation error: {exc}") from exc
