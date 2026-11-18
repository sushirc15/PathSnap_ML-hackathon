#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Starting PathSnap FastAPI backend on 0.0.0.0:8000 (Laptop 1)..."
exec .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --access-log --log-level info
