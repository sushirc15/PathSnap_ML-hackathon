"""FastAPI application entrypoint for PathSnap (Laptop 1 backend)."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import get_service, router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("pathsnap.api")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        svc = get_service()
        info = svc.get_health_info()
        logger.info(
            "PathSnap backend ready | model=%s | partition=%s | trips=%d",
            info["model_name"],
            info["active_partition"],
            info["available_trip_count"],
        )
    except Exception as exc:
        logger.warning("Deferred model initialization on startup: %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="PathSnap Distributed Map-Matching API",
        description=(
            "Distributed map-matching and route reconstruction backend hosting the trained "
            "candidate classifier, spatial index, directed road-network HMM/Viterbi decoder, "
            "and isolated server-side evaluation."
        ),
        version="1.0.0",
        lifespan=_lifespan,
    )

    cors_env = os.getenv("CORS_ORIGINS", "*")
    allow_origins = [o.strip() for o in cors_env.split(",") if o.strip()] or ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        client_host = request.client.host if request.client else "unknown"
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "HTTP %s %s from=%s status=%d duration_ms=%.1f",
            request.method,
            request.url.path,
            client_host,
            response.status_code,
            elapsed_ms,
        )
        return response

    app.include_router(router)
    return app


app = create_app()
