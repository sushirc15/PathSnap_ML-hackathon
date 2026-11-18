#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../frontend"
echo "Starting PathSnap Evaluator Dashboard (Laptop 2)..."
echo "Using VITE_API_BASE_URL=${VITE_API_BASE_URL:-http://127.0.0.1:8000}"
exec npm run dev -- --host 0.0.0.0 --port 5173
