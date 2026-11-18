"""Folium interactive map generation for PathSnap EDA, model comparison, and evaluator views."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import folium
import numpy as np

from src.data.preprocessing import PreparedTrip


def _extract_local_road_segments(
    trip: PreparedTrip,
    buffer_m: float = 200.0,
    max_arcs: int = 2500,
) -> List[List[Tuple[float, float]]]:
    """Return local road-network polylines around the GPS trajectory bounding box."""
    min_x = float(np.min(trip.gps_x)) - buffer_m
    max_x = float(np.max(trip.gps_x)) + buffer_m
    min_y = float(np.min(trip.gps_y)) - buffer_m
    max_y = float(np.max(trip.gps_y)) + buffer_m

    arc_ids = trip.spatial_index.query_arcs_in_bbox(min_x, min_y, max_x, max_y)
    if len(arc_ids) > max_arcs:
        # Subsample evenly to keep HTML map crisp and responsive
        step = max(1, len(arc_ids) // max_arcs)
        arc_ids = arc_ids[::step]
    return trip.network.get_arc_lonlat_coords(arc_ids)


def _merge_arc_polylines(
    trip: PreparedTrip, arc_ids: Sequence[int]
) -> List[List[Tuple[float, float]]]:
    """Merge consecutive topologically connected arcs into clean contiguous polylines."""
    if len(arc_ids) == 0:
        return []
    net = trip.network
    polylines: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    prev_to: Optional[int] = None
    prev_arc: Optional[int] = None

    for aid in arc_ids:
        aid_int = int(aid)
        if aid_int < 0 or aid_int >= net.num_arcs:
            continue
        if prev_arc is not None and aid_int == prev_arc:
            continue
        u = int(net.from_nodes[aid_int])
        v = int(net.to_nodes[aid_int])
        p_u = (float(net.nodes_lonlat[u, 1]), float(net.nodes_lonlat[u, 0]))
        p_v = (float(net.nodes_lonlat[v, 1]), float(net.nodes_lonlat[v, 0]))
        if prev_to is not None and u == prev_to and current:
            current.append(p_v)
        else:
            if current:
                polylines.append(current)
            current = [p_u, p_v]
        prev_to = v
        prev_arc = aid_int

    if current:
        polylines.append(current)
    return polylines


def build_trip_folium_map(
    trip: PreparedTrip,
    predicted_arcs: Optional[Sequence[int]] = None,
    classifier_only_arcs: Optional[Sequence[int]] = None,
    reference_arcs: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    show_road_network: bool = True,
    max_gps_markers: int = 800,
) -> folium.Map:
    """Create an interactive Folium map with distinguishable layers, legend, and layer control.

    Note: `reference_arcs` must ONLY be passed for offline EDA / development / report maps,
    NEVER for evaluator-facing API maps.
    """
    center_lat = float(np.median(trip.lat))
    center_lon = float(np.median(trip.lon))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri Street",
    ).add_to(m)

    # Fit bounds tightly around trajectory
    sw = [float(np.min(trip.lat)), float(np.min(trip.lon))]
    ne = [float(np.max(trip.lat)), float(np.max(trip.lon))]
    m.fit_bounds([sw, ne], padding=(25, 25))

    legend_items: List[Tuple[str, str]] = []

    # 1. Local road network geometry
    if show_road_network:
        road_group = folium.FeatureGroup(name="Road Network (Local)", show=True)
        road_lines = _extract_local_road_segments(trip, buffer_m=150.0, max_arcs=2000)
        if road_lines:
            folium.PolyLine(
                locations=road_lines,
                color="#94a3b8",
                weight=1.4,
                opacity=0.45,
            ).add_to(road_group)
        road_group.add_to(m)
        legend_items.append(("#94a3b8", "Road Network (Local)"))

    # 2. Reference route (offline/EDA/dev only)
    if reference_arcs is not None and len(reference_arcs) > 0:
        ref_group = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        for line in _merge_arc_polylines(trip, reference_arcs):
            folium.PolyLine(
                locations=line,
                color="#16a34a",
                weight=6,
                opacity=0.75,
                tooltip="Reference Route",
            ).add_to(ref_group)
        ref_group.add_to(m)
        legend_items.append(("#16a34a", "Reference Route (.route)"))

    # 3. Classifier-only predicted route (optional comparison)
    if classifier_only_arcs is not None and len(classifier_only_arcs) > 0:
        clf_group = folium.FeatureGroup(name="Classifier-Only Prediction", show=False)
        for line in _merge_arc_polylines(trip, classifier_only_arcs):
            folium.PolyLine(
                locations=line,
                color="#f97316",
                weight=3.5,
                opacity=0.8,
                dash_array="6, 6",
                tooltip="Classifier-Only Route",
            ).add_to(clf_group)
        clf_group.add_to(m)
        legend_items.append(("#f97316", "Classifier-Only Prediction"))

    # 4. HMM/Viterbi predicted route
    if predicted_arcs is not None and len(predicted_arcs) > 0:
        pred_group = folium.FeatureGroup(name="HMM/Viterbi Predicted Route", show=True)
        for line in _merge_arc_polylines(trip, predicted_arcs):
            folium.PolyLine(
                locations=line,
                color="#dc2626",
                weight=3.5,
                opacity=0.92,
                tooltip="HMM/Viterbi Predicted Route",
            ).add_to(pred_group)
        pred_group.add_to(m)
        legend_items.append(("#dc2626", "HMM/Viterbi Predicted Route"))

    # 5. Raw noisy GPS observations
    gps_group = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    gps_coords = list(zip(trip.lat.tolist(), trip.lon.tolist()))
    folium.PolyLine(
        locations=gps_coords,
        color="#2563eb",
        weight=1.5,
        opacity=0.45,
    ).add_to(gps_group)

    step = max(1, len(trip.lat) // max_gps_markers)
    for idx in range(0, len(trip.lat), step):
        folium.CircleMarker(
            location=[float(trip.lat[idx]), float(trip.lon[idx])],
            radius=2.5,
            color="#1d4ed8",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.85,
            popup=f"Obs #{idx} | t={trip.timestamp[idx]:.1f}s | v={trip.speed_smooth[idx]:.1f} m/s",
        ).add_to(gps_group)
    gps_group.add_to(m)
    legend_items.append(("#2563eb", f"Raw GPS Points (n={len(trip.lat)})"))

    # Start & End markers
    markers_group = folium.FeatureGroup(name="Start / End Markers", show=True)
    folium.Marker(
        location=[float(trip.lat[0]), float(trip.lon[0])],
        popup=f"Start (t={trip.timestamp[0]:.1f}s)",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(markers_group)
    folium.Marker(
        location=[float(trip.lat[-1]), float(trip.lon[-1])],
        popup=f"End (t={trip.timestamp[-1]:.1f}s)",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(markers_group)
    markers_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Title banner & legend overlay
    map_title = title or f"PathSnap Map — Trip {trip.trip_id} ({trip.partition})"
    legend_rows = "".join(
        f'<div style="display:flex;align-items:center;margin:3px 0;">'
        f'<span style="display:inline-block;width:16px;height:4px;background:{col};margin-right:8px;border-radius:2px;"></span>'
        f'<span style="font-size:12px;color:#1e293b;">{label}</span></div>'
        for col, label in legend_items
    )
    overlay_html = f"""
    <div style="position:fixed;top:12px;left:56px;z-index:9999;background:rgba(255,255,255,0.94);
                padding:8px 14px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.15);
                font-family:system-ui,-apple-system,sans-serif;border:1px solid #e2e8f0;">
        <div style="font-weight:700;font-size:14px;color:#0f172a;">{map_title}</div>
    </div>
    <div style="position:fixed;bottom:24px;left:14px;z-index:9999;background:rgba(255,255,255,0.94);
                padding:10px 14px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.15);
                font-family:system-ui,-apple-system,sans-serif;border:1px solid #e2e8f0;">
        <div style="font-weight:700;font-size:12px;color:#0f172a;margin-bottom:5px;">Map Legend</div>
        {legend_rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(overlay_html))
    return m


def save_trip_folium_map(m: folium.Map, output_path: Path) -> Path:
    """Save Folium map HTML to disk, creating parent directories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    return output_path
