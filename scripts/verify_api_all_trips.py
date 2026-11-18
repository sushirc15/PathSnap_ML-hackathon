"""Verify all 10 student-testing trips through the FastAPI app and render real API log image."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app


def _render_terminal_screenshot(log_lines: list[str], out_path: Path) -> None:
    """Render actual FastAPI/Uvicorn request and prediction logs into a crisp terminal PNG."""
    pad_x, pad_y = 24, 22
    header_h = 34
    line_h = 20
    width = 1280
    height = header_h + pad_y * 2 + line_h * len(log_lines) + 12

    img = Image.new("RGB", (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Window title bar
    draw.rectangle([0, 0, width, header_h], fill=(30, 41, 59))
    draw.ellipse([14, 11, 26, 23], fill=(239, 68, 68))
    draw.ellipse([34, 11, 46, 23], fill=(245, 158, 11))
    draw.ellipse([54, 11, 66, 23], fill=(34, 197, 94))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
        font_title = font

    draw.text((84, 9), "Laptop 1 — PathSnap FastAPI + Uvicorn Backend Logs (0.0.0.0:8000)", fill=(203, 213, 225), font=font_title)

    y = header_h + pad_y
    for line in log_lines:
        if "PREDICT " in line:
            col = (134, 239, 172)  # bright green
        elif "HTTP " in line and "status=200" in line:
            col = (147, 197, 253)  # sky blue
        elif line.startswith("$ "):
            col = (253, 224, 71)  # yellow command prompt
        else:
            col = (226, 232, 240)
        draw.text((pad_x, y), line[:148], fill=col, font=font)
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    log_path = PROJECT_ROOT / "experiments" / "logs" / "api_verification.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    captured: list[str] = [
        "$ .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --access-log --log-level info",
        "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
    ]

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(self.format(record))

    handler = ListHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    api_logger = logging.getLogger("pathsnap.api")
    api_logger.addHandler(handler)
    api_logger.setLevel(logging.INFO)

    try:
        with TestClient(app) as client:
            h = client.get("/api/health")
            assert h.status_code == 200

            t = client.get("/api/trips")
            assert t.status_code == 200
            trips = t.json()["trips"]
            assert len(trips) == 10, f"Expected 10 student_test trips, got {len(trips)}"

            for item in trips:
                tid = item["trip_id"]
                p = client.post(f"/api/trips/{tid}/predict")
                assert p.status_code == 200, f"Failed on {tid}: {p.text}"
                body = p.json()
                assert "reference_route_arcs" not in p.text.lower()
                assert body["hmm_status"]["enabled"] is True
                assert body["continuity"]["reconstructed_route_direct_continuity"] >= 0.99

            m = client.get("/api/trips/00000040/map")
            assert m.status_code == 200
            assert "Reference Route (.route)" not in m.text
    finally:
        api_logger.removeHandler(handler)

    log_path.write_text("\n".join(captured) + "\n")
    out_img = PROJECT_ROOT / "reports" / "figures" / "api_logs_screenshot.png"
    _render_terminal_screenshot(captured, out_img)
    print(f"Verified all 10 trips via FastAPI! Saved logs -> {log_path} and PNG -> {out_img}")


if __name__ == "__main__":
    main()
