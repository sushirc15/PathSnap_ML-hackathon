import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { PredictionResponse } from "../types";

interface RouteMapProps {
  prediction: PredictionResponse | null;
  loading: boolean;
}

export function RouteMap({ prediction, loading }: RouteMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerGroupRef = useRef<L.LayerGroup | null>(null);

  const [showRoadNet, setShowRoadNet] = useState(true);
  const [showGps, setShowGps] = useState(true);
  const [showPredicted, setShowPredicted] = useState(true);

  // Initialize Leaflet map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: [49.85, 8.6],
      zoom: 5,
      zoomControl: true,
    });

    const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 19,
    });
    const carto = L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      {
        attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
        maxZoom: 19,
      }
    );

    carto.addTo(map);
    L.control.layers({ "Carto Light": carto, OpenStreetMap: osm }).addTo(map);
    L.control.scale({ imperial: false }).addTo(map);

    layerGroupRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
      layerGroupRef.current = null;
    };
  }, []);

  // Render prediction geometry layers (STRICTLY evaluator-safe: no ground truth)
  useEffect(() => {
    const map = mapRef.current;
    const group = layerGroupRef.current;
    if (!map || !group) return;

    group.clearLayers();
    if (!prediction) return;

    const { geometry } = prediction;

    // 1. Local road network geometry
    if (showRoadNet && geometry.local_road_segments.length > 0) {
      L.polyline(geometry.local_road_segments, {
        color: "#94a3b8",
        weight: 1.4,
        opacity: 0.48,
      }).addTo(group);
    }

    // 2. HMM/Viterbi predicted road route
    if (showPredicted && geometry.predicted_polylines.length > 0) {
      for (const line of geometry.predicted_polylines) {
        L.polyline(line, {
          color: "#dc2626",
          weight: 4.2,
          opacity: 0.92,
        })
          .bindTooltip("HMM/Viterbi Predicted Route")
          .addTo(group);
      }
    }

    // 3. Raw noisy GPS observations
    if (showGps && geometry.gps_points.length > 0) {
      const gpsLatLngs = geometry.gps_points.map((p) => [p.lat, p.lon] as [number, number]);
      L.polyline(gpsLatLngs, {
        color: "#2563eb",
        weight: 1.6,
        opacity: 0.42,
      }).addTo(group);

      for (const p of geometry.gps_points) {
        L.circleMarker([p.lat, p.lon], {
          radius: 2.6,
          color: "#1d4ed8",
          fillColor: "#3b82f6",
          fillOpacity: 0.85,
          weight: 1,
        })
          .bindPopup(`GPS Obs #${p.idx}<br/>t = ${p.t}s`)
          .addTo(group);
      }
    }

    // 4. Start & End markers
    const startIcon = L.divIcon({
      className: "pathsnap-pin",
      html: `<div class="pin-badge pin-start">START</div>`,
      iconSize: [52, 22],
      iconAnchor: [26, 11],
    });
    const endIcon = L.divIcon({
      className: "pathsnap-pin",
      html: `<div class="pin-badge pin-end">END</div>`,
      iconSize: [42, 22],
      iconAnchor: [21, 11],
    });

    L.marker(geometry.start, { icon: startIcon }).addTo(group);
    L.marker(geometry.end, { icon: endIcon }).addTo(group);

    map.fitBounds(geometry.bounds, { padding: [36, 36] });
  }, [prediction, showRoadNet, showGps, showPredicted]);

  return (
    <div className="map-card">
      <div className="map-toolbar">
        <div className="map-toolbar-title">
          {prediction ? (
            <>
              <strong>Trip {prediction.trip_id}</strong>
              <span className="muted-tag">{prediction.prediction.utm_crs}</span>
            </>
          ) : (
            <strong>Interactive Route Map</strong>
          )}
        </div>
        <div className="layer-toggles">
          <label className="toggle-pill">
            <input
              type="checkbox"
              checked={showGps}
              onChange={(e) => setShowGps(e.target.checked)}
            />
            <span className="swatch swatch-gps" />
            Raw GPS
          </label>
          <label className="toggle-pill">
            <input
              type="checkbox"
              checked={showPredicted}
              onChange={(e) => setShowPredicted(e.target.checked)}
            />
            <span className="swatch swatch-pred" />
            Predicted Route
          </label>
          <label className="toggle-pill">
            <input
              type="checkbox"
              checked={showRoadNet}
              onChange={(e) => setShowRoadNet(e.target.checked)}
            />
            <span className="swatch swatch-road" />
            Road Network
          </label>
        </div>
      </div>

      <div className="map-canvas-wrap">
        <div ref={containerRef} className="leaflet-canvas" />
        {loading && (
          <div className="map-overlay">
            <div className="spinner" />
            <span>Running Spatial Indexing, Classifier &amp; HMM/Viterbi...</span>
          </div>
        )}
        {!loading && !prediction && (
          <div className="map-empty-hint">
            Select a trip from the left panel to execute live map-matching on Laptop 1.
          </div>
        )}
        <div className="map-legend">
          <div className="legend-title">Legend</div>
          <div className="legend-row">
            <span className="swatch swatch-gps" /> Raw GPS Observations
          </div>
          <div className="legend-row">
            <span className="swatch swatch-pred" /> HMM/Viterbi Predicted Route
          </div>
          <div className="legend-row">
            <span className="swatch swatch-road" /> Local Road Network
          </div>
          <div className="legend-row">
            <span className="dot-start" /> Start &nbsp;/&nbsp; <span className="dot-end" /> End
          </div>
        </div>
      </div>
    </div>
  );
}
