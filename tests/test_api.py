"""FastAPI endpoint and ground-truth separation tests for PathSnap backend."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_api_health_trips_predict_and_map() -> None:
    with TestClient(app) as client:
        # 1. Health check
        h_res = client.get("/api/health")
        assert h_res.status_code == 200
        h_data = h_res.json()
        assert h_data["status"] == "ok"
        assert h_data["hmm_enabled"] is True
        assert h_data["available_trip_count"] >= 1

        # 2. Trips listing (dynamically loaded, not hardcoded)
        t_res = client.get("/api/trips")
        assert t_res.status_code == 200
        t_data = t_res.json()
        assert t_data["trip_count"] == h_data["available_trip_count"]
        assert len(t_data["trips"]) >= 1

        # Pick shortest trip for fast CI test
        shortest = min(t_data["trips"], key=lambda x: x["n_gps_observations"])
        tid = shortest["trip_id"]

        # 3. Prediction endpoint
        p_res = client.post(f"/api/trips/{tid}/predict")
        assert p_res.status_code == 200
        p_data = p_res.json()
        assert p_data["trip_id"] == tid
        assert p_data["hmm_status"]["enabled"] is True
        assert p_data["prediction"]["observation_count"] == shortest["n_gps_observations"]
        assert len(p_data["prediction"]["predicted_route_arc_ids"]) > 0
        assert 0.0 <= p_data["continuity"]["reconstructed_route_direct_continuity"] <= 1.0

        # STRICT GROUND-TRUTH SEPARATION CHECK:
        # Response must NEVER contain raw reference arc IDs or reference route geometry
        raw_text = p_res.text.lower()
        assert "reference_route_arcs" not in raw_text
        assert "reference_polylines" not in raw_text
        assert "ground_truth_arcs" not in raw_text

        # Aggregate evaluation metrics present when reference exists
        ev = p_data["evaluation"]
        if ev["ground_truth_available"]:
            assert 0.0 <= ev["top1_accuracy"] <= 1.0
            assert 0.0 <= ev["macro_f1"] <= 1.0

        # 4. Evaluator-safe Folium map endpoint (must NOT include Reference Route layer)
        m_res = client.get(f"/api/trips/{tid}/map")
        assert m_res.status_code == 200
        assert "text/html" in m_res.headers["content-type"]
        assert "Reference Route (.route)" not in m_res.text

        # 5. Missing trip returns 404 cleanly
        bad_res = client.post("/api/trips/nonexistent_99999/predict")
        assert bad_res.status_code == 404
