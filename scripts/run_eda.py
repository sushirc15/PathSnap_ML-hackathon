"""Run complete dataset audit, inventory generation, EDA charts, and EDA Folium maps."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.loader import list_trip_ids, load_trip_inputs, load_trip_route
from src.data.preprocessing import prepare_trip
from src.data.validation import validate_trip
from src.visualization.folium_maps import build_trip_folium_map, save_trip_folium_map


def main() -> None:
    t0 = time.perf_counter()
    cfg = get_config()

    partitions = [
        ("train", cfg.train_dir),
        ("student_test", cfg.student_test_dir),
        ("evaluator_test", cfg.evaluator_test_dir),
    ]

    rows = []
    all_warnings = []
    all_errors = []
    dt_samples_train = []
    step_dist_samples_train = []
    sample_trip_id = None

    for part_name, part_dir in partitions:
        tids = list_trip_ids(part_dir)
        print(f"[{part_name}] Found {len(tids)} trips in {part_dir}")
        for idx, tid in enumerate(tids):
            trip = load_trip_inputs(part_dir, tid, partition=part_name)
            route = load_trip_route(part_dir, tid)
            report = validate_trip(trip, route=route, include_graph_stats=True)
            rows.append(report.metrics)
            for w in report.warnings:
                all_warnings.append({"partition": part_name, "trip_id": tid, "warning": w})
            for e in report.errors:
                all_errors.append({"partition": part_name, "trip_id": tid, "error": e})

            if part_name == "train":
                if sample_trip_id is None:
                    sample_trip_id = tid
                ts = trip.track["timestamp"].to_numpy(dtype=np.float64)
                if len(ts) > 1:
                    dt_samples_train.append(np.diff(ts))
                # Step distances computed in UTM
                from src.geo.projection import TripProjector
                proj = TripProjector.from_lonlat(
                    trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy()
                )
                gx, gy = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
                if len(gx) > 1:
                    step_dist_samples_train.append(np.hypot(np.diff(gx), np.diff(gy)))

            if (idx + 1) % 20 == 0 or (idx + 1) == len(tids):
                print(f"  audited {idx + 1}/{len(tids)} ({part_name})")

    inv_df = pd.DataFrame(rows)
    inv_path = cfg.experiments_dir / "dataset_inventory.csv"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(inv_path, index=False)
    print(f"Saved dataset inventory -> {inv_path}")

    # Build partition summary
    summary = {"partitions": {}, "warnings_count": len(all_warnings), "errors_count": len(all_errors)}
    for part_name in ("train", "student_test", "evaluator_test"):
        sub = inv_df[inv_df["partition"] == part_name] if len(inv_df) > 0 else pd.DataFrame()
        if len(sub) == 0:
            summary["partitions"][part_name] = {"n_trips": 0, "status": "not_supplied"}
            continue
        summary["partitions"][part_name] = {
            "n_trips": int(len(sub)),
            "total_gps_obs": int(sub["n_gps"].sum()),
            "gps_obs_min": int(sub["n_gps"].min()),
            "gps_obs_median": float(sub["n_gps"].median()),
            "gps_obs_max": int(sub["n_gps"].max()),
            "total_nodes_mean": float(sub["n_nodes"].mean()),
            "total_nodes_min": int(sub["n_nodes"].min()),
            "total_nodes_max": int(sub["n_nodes"].max()),
            "total_arcs_mean": float(sub["n_arcs"].mean()),
            "total_arcs_min": int(sub["n_arcs"].min()),
            "total_arcs_max": int(sub["n_arcs"].max()),
            "missing_values_total": int(sub["track_na"].sum() + sub["nodes_na"].sum() + sub["arcs_na"].sum()),
            "invalid_coords_total": int(sub["invalid_gps_coords"].sum() + sub["invalid_node_coords"].sum()),
            "invalid_arc_refs_total": int(sub["invalid_arc_refs"].sum()),
            "non_monotonic_ts_total": int(sub["non_monotonic_ts"].sum()),
            "duplicate_ts_total": int(sub["duplicate_ts"].sum()),
            "mean_pct_1s_steps": float(sub["pct_1s_steps"].mean()),
            "trips_with_gt5s_gaps": int((sub["gt_5s_gaps"] > 0).sum()),
            "trips_with_gt30s_gaps": int((sub["gt_30s_gaps"] > 0).sum()),
            "max_dt_s_overall": float(sub["max_dt_s"].max()),
            "max_gps_jump_m_overall": float(sub["max_gps_jump_m"].max()),
            "lon_range": [float(sub["lon_min"].min()), float(sub["lon_max"].max())],
            "lat_range": [float(sub["lat_min"].min()), float(sub["lat_max"].max())],
            "utm_epsg_codes": sorted(int(x) for x in sub["utm_epsg"].unique()),
            "mean_isolated_nodes": float(sub["isolated_nodes"].mean()),
            "mean_wcc": float(sub["weakly_connected_components"].mean()),
            "mean_scc": float(sub["strongly_connected_components"].mean()),
            "mean_arc_len_m": float(sub["mean_arc_len_m"].mean()),
            "median_arc_len_m": float(sub["median_arc_len_m"].median()),
            "mean_pct_two_way": float(sub["pct_two_way_arcs"].mean()),
            "trips_with_route": int(sub["has_route"].sum()),
            "mean_route_arcs": float(sub["n_route_arcs"].mean()),
            "mean_route_direct_continuity": float(sub["route_direct_continuity"].mean()),
            "total_route_breaks": int(sub["route_breaks"].sum()),
            "trips_with_repeated_route_arcs": int(sub["has_repeated_route_arcs"].sum()),
        }

    summary["sample_warnings"] = all_warnings[:20]
    summary["errors"] = all_errors

    summary_path = cfg.experiments_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved EDA summary -> {summary_path}")
    print(json.dumps(summary["partitions"], indent=2))

    # --- Generate EDA Figures ---
    fig_dir = cfg.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_df = inv_df[inv_df["partition"] == "train"]
    stest_df = inv_df[inv_df["partition"] == "student_test"]

    # Figure 1: Dataset overview (GPS counts, Route lengths, Road network size, Geographic footprint)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)

    axes[0, 0].hist(train_df["n_gps"], bins=20, color="#2563eb", alpha=0.78, label="Train (80)")
    axes[0, 0].hist(stest_df["n_gps"], bins=10, color="#f97316", alpha=0.85, label="Student Test (10)")
    axes[0, 0].set_title("GPS Observations per Trip", fontweight="bold")
    axes[0, 0].set_xlabel("Number of GPS Observations")
    axes[0, 0].set_ylabel("Trip Count")
    axes[0, 0].legend()

    axes[0, 1].scatter(train_df["n_nodes"] / 1e3, train_df["n_arcs"] / 1e3, c="#0ea5e9", alpha=0.8, s=45, label="Train")
    axes[0, 1].scatter(stest_df["n_nodes"] / 1e3, stest_df["n_arcs"] / 1e3, c="#ea580c", marker="D", s=55, label="Student Test")
    axes[0, 1].set_title("Per-Trip Road Network Size", fontweight="bold")
    axes[0, 1].set_xlabel("Nodes (thousands)")
    axes[0, 1].set_ylabel("Directed Arcs (thousands)")
    axes[0, 1].legend()

    axes[1, 0].scatter(train_df["n_gps"], train_df["n_route_arcs"], c="#16a34a", alpha=0.75, s=45, label="Train")
    axes[1, 0].scatter(stest_df["n_gps"], stest_df["n_route_arcs"], c="#dc2626", marker="^", s=55, label="Student Test")
    axes[1, 0].set_title("GPS Observations vs. Reference Route Arcs", fontweight="bold")
    axes[1, 0].set_xlabel("Number of GPS Observations (.track)")
    axes[1, 0].set_ylabel("Number of Route Arcs (.route)")
    axes[1, 0].legend()

    # Trip geographic centroids
    train_lon_mid = 0.5 * (train_df["lon_min"] + train_df["lon_max"])
    train_lat_mid = 0.5 * (train_df["lat_min"] + train_df["lat_max"])
    stest_lon_mid = 0.5 * (stest_df["lon_min"] + stest_df["lon_max"])
    stest_lat_mid = 0.5 * (stest_df["lat_min"] + stest_df["lat_max"])
    axes[1, 1].scatter(train_lon_mid, train_lat_mid, c="#2563eb", alpha=0.75, s=45, label="Train (80)")
    axes[1, 1].scatter(stest_lon_mid, stest_lat_mid, c="#dc2626", marker="*", s=110, label="Student Test (10)")
    axes[1, 1].set_title("Geographical Distribution of Trip Centroids (WGS84)", fontweight="bold")
    axes[1, 1].set_xlabel("Longitude (°)")
    axes[1, 1].set_ylabel("Latitude (°)")
    axes[1, 1].legend()

    fig.suptitle("PathSnap Dataset Exploratory Overview (Train & Student-Test Partitions)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "eda_overview.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Sampling intervals, GPS step displacements, and road segment lengths
    all_dt = np.concatenate(dt_samples_train)
    all_step = np.concatenate(step_dist_samples_train)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)

    dt_clipped = np.clip(all_dt, 0, 15)
    axes[0].hist(dt_clipped, bins=30, color="#3b82f6", edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_title("GPS Sampling Intervals (Train, capped at 15s)", fontweight="bold")
    axes[0].set_xlabel("Δt (seconds)")
    axes[0].set_ylabel("Count (log scale)")

    step_clipped = np.clip(all_step, 0, 60)
    axes[1].hist(step_clipped, bins=35, color="#10b981", edgecolor="white")
    axes[1].set_title("Consecutive GPS Displacement (Train, ≤60m)", fontweight="bold")
    axes[1].set_xlabel("Displacement (meters)")
    axes[1].set_ylabel("Count")

    axes[2].hist(train_df["max_gps_jump_m"], bins=20, color="#ef4444", alpha=0.8, edgecolor="white")
    axes[2].set_title("Max Single-Step GPS Jump per Trip (Train)", fontweight="bold")
    axes[2].set_xlabel("Max GPS Jump (meters)")
    axes[2].set_ylabel("Trip Count")

    fig.suptitle("GPS Temporal Sampling & Spatial Jump Characteristics (Training Partition)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_dir / "eda_sampling_and_gaps.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Sample GPS trajectory vs road network & reference route (matplotlib + Folium)
    if sample_trip_id is not None:
        raw_t = load_trip_inputs(cfg.train_dir, sample_trip_id, partition="train")
        route_t = load_trip_route(cfg.train_dir, sample_trip_id)
        prep_t = prepare_trip(raw_t)

        # Static matplotlib plot of local road network, GPS track, and reference route
        fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
        pad = 250.0
        min_x, max_x = prep_t.gps_x.min() - pad, prep_t.gps_x.max() + pad
        min_y, max_y = prep_t.gps_y.min() - pad, prep_t.gps_y.max() + pad
        local_arcs = prep_t.spatial_index.query_arcs_in_bbox(min_x, min_y, max_x, max_y)
        from matplotlib.collections import LineCollection
        net = prep_t.network
        if len(local_arcs) > 0:
            segs = np.stack(
                [
                    np.column_stack([net.x1[local_arcs], net.y1[local_arcs]]),
                    np.column_stack([net.x2[local_arcs], net.y2[local_arcs]]),
                ],
                axis=1,
            )
            ax.add_collection(LineCollection(segs, colors="#cbd5e1", linewidths=0.7, alpha=0.8, label="Local Road Network"))
        if route_t is not None and len(route_t) > 0:
            r_segs = np.stack(
                [
                    np.column_stack([net.x1[route_t], net.y1[route_t]]),
                    np.column_stack([net.x2[route_t], net.y2[route_t]]),
                ],
                axis=1,
            )
            ax.add_collection(LineCollection(r_segs, colors="#16a34a", linewidths=2.8, alpha=0.85, label="Reference Route (.route)"))
        ax.scatter(prep_t.gps_x, prep_t.gps_y, s=6, c="#2563eb", alpha=0.75, zorder=5, label="Raw GPS Observations")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect("equal")
        ax.set_title(f"Trip {sample_trip_id} (Train): GPS Observations, Local Road Network & Reference Route", fontweight="bold")
        ax.set_xlabel(f"UTM Easting (m, EPSG:{prep_t.projector.epsg_code})")
        ax.set_ylabel("UTM Northing (m)")
        ax.legend(loc="best")
        fig.savefig(fig_dir / "eda_sample_trajectory.png", bbox_inches="tight")
        plt.close(fig)

        # Interactive Folium EDA map
        fmap = build_trip_folium_map(
            prep_t,
            reference_arcs=route_t,
            title=f"EDA Map — Training Trip {sample_trip_id} (Raw GPS, Local Road Network & Reference Route)",
            show_road_network=True,
        )
        map_path = save_trip_folium_map(fmap, cfg.reports_dir / "maps" / f"eda_trip_{sample_trip_id}.html")
        print(f"Saved Folium EDA map -> {map_path}")

    print(f"EDA completed in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
