"""Reusable publication-quality plotting utilities for PathSnap reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from src.data.preprocessing import PreparedTrip


def save_static_route_comparison_figure(
    trip: PreparedTrip,
    predicted_arcs: Sequence[int],
    classifier_only_arcs: Optional[Sequence[int]] = None,
    reference_arcs: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    zoom_box: Optional[tuple[float, float, float, float]] = None,
) -> Path:
    """Render a crisp high-DPI geographic comparison figure matching the Folium map layers.

    Used to embed clear static exports alongside interactive Folium HTML maps in the PDF report.
    """
    net = trip.network
    pad = 180.0
    if zoom_box is not None:
        min_x, max_x, min_y, max_y = zoom_box
    else:
        min_x, max_x = float(trip.gps_x.min()) - pad, float(trip.gps_x.max()) + pad
        min_y, max_y = float(trip.gps_y.min()) - pad, float(trip.gps_y.max()) + pad

    local_arcs = trip.spatial_index.query_arcs_in_bbox(min_x, min_y, max_x, max_y)

    fig, ax = plt.subplots(figsize=(9.5, 6.8), dpi=180)
    ax.set_facecolor("#f8fafc")

    if len(local_arcs) > 0:
        segs = np.stack(
            [
                np.column_stack([net.x1[local_arcs], net.y1[local_arcs]]),
                np.column_stack([net.x2[local_arcs], net.y2[local_arcs]]),
            ],
            axis=1,
        )
        ax.add_collection(
            LineCollection(segs, colors="#cbd5e1", linewidths=0.75, alpha=0.85, label="Road Network", zorder=1)
        )

    if reference_arcs is not None and len(reference_arcs) > 0:
        r_arr = np.asarray([a for a in reference_arcs if a >= 0], dtype=np.int32)
        r_segs = np.stack(
            [
                np.column_stack([net.x1[r_arr], net.y1[r_arr]]),
                np.column_stack([net.x2[r_arr], net.y2[r_arr]]),
            ],
            axis=1,
        )
        ax.add_collection(
            LineCollection(r_segs, colors="#16a34a", linewidths=4.2, alpha=0.65, label="Reference Route (.route)", zorder=2)
        )

    if classifier_only_arcs is not None and len(classifier_only_arcs) > 0:
        c_arr = np.asarray([a for a in classifier_only_arcs if a >= 0], dtype=np.int32)
        c_segs = np.stack(
            [
                np.column_stack([net.x1[c_arr], net.y1[c_arr]]),
                np.column_stack([net.x2[c_arr], net.y2[c_arr]]),
            ],
            axis=1,
        )
        ax.add_collection(
            LineCollection(
                c_segs,
                colors="#f97316",
                linewidths=2.0,
                linestyles="dashed",
                alpha=0.85,
                label="Classifier-Only Prediction",
                zorder=3,
            )
        )

    if len(predicted_arcs) > 0:
        p_arr = np.asarray([a for a in predicted_arcs if a >= 0], dtype=np.int32)
        p_segs = np.stack(
            [
                np.column_stack([net.x1[p_arr], net.y1[p_arr]]),
                np.column_stack([net.x2[p_arr], net.y2[p_arr]]),
            ],
            axis=1,
        )
        ax.add_collection(
            LineCollection(p_segs, colors="#dc2626", linewidths=2.3, alpha=0.92, label="HMM/Viterbi Predicted Route", zorder=4)
        )

    ax.scatter(trip.gps_x, trip.gps_y, s=7, c="#2563eb", alpha=0.72, label="Raw GPS Observations", zorder=5)
    ax.scatter([trip.gps_x[0]], [trip.gps_y[0]], s=75, c="#16a34a", edgecolor="white", linewidth=1.5, marker="o", zorder=6, label="Start")
    ax.scatter([trip.gps_x[-1]], [trip.gps_y[-1]], s=75, c="#dc2626", edgecolor="white", linewidth=1.5, marker="s", zorder=6, label="End")

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.45, color="#94a3b8")
    ax.set_xlabel(f"UTM Easting (m, EPSG:{trip.projector.epsg_code})")
    ax.set_ylabel("UTM Northing (m)")
    ax.set_title(title or f"PathSnap Map-Matching Comparison — Trip {trip.trip_id}", fontweight="bold", fontsize=11.5)
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

    out = output_path or Path(f"reports/figures/map_comparison_{trip.trip_id}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
