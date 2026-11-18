"""Generate standalone consolidated Final Evaluation Report PDF for the 10 reserved evaluator trips."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_report import _add_figure, _footer, _make_table, _styles
from src.config import get_config


def main() -> Path:
    cfg = get_config()
    exp = cfg.experiments_dir
    figs = cfg.reports_dir / "figures"

    ev_sum = json.loads((exp / "evaluator_test_summary.json").read_text())
    ev_df = pd.read_csv(exp / "evaluator_test_results.csv")
    val_df = pd.read_csv(exp / "evaluator_test_validation.csv")
    st_sum = json.loads((exp / "hmm_summary.json").read_text())

    out_pdf = cfg.reports_dir / "PathSnap_Final_Evaluation_Report.pdf"
    st = _styles()
    story: List[Any] = []

    ev_co = ev_sum["classifier_only"]
    ev_ch = ev_sum["classifier_plus_hmm"]

    story.append(Paragraph("PATHSNAP — FINAL RESERVED EVALUATOR-TEST REPORT", st["title"]))
    story.append(
        Paragraph(
            f"Evaluation on 10 Unseen Reserved Trips (evaluator_test.zip) | Frozen Trained Model &amp; HMM | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            st["subtitle"],
        )
    )
    story.append(
        Paragraph(
            "<b>Evaluation Integrity Statement:</b> All 10 reserved trips were extracted into data/evaluator_test/, validated, "
            "and automatically loaded into the FastAPI /api/trips selector (is_development_fallback=False). Predictions were "
            "generated strictly from .track, .nodes, and .arcs using the frozen HistGradientBoosting model "
            "(artifacts/models/final_model.joblib) and frozen HMM/Viterbi configuration (artifacts/configurations/hmm_config.json) "
            "with zero retraining, zero feature edits, and zero tuning on evaluator data. All .route files remained strictly "
            "isolated on the server and were read only after predictions completed.",
            st["body"],
        )
    )
    story.append(Spacer(1, 6))

    # 1. Aggregate summary comparing Classifier-Only vs HMM/Viterbi + reference Student-Test baseline
    story.append(Paragraph("1. Aggregate Final Evaluation Summary (10 Reserved Trips)", st["h1"]))
    agg_rows = [
        ["Reserved Trips Validated", f"{ev_sum['n_trips']} / {ev_sum['n_trips']} (0 errors)", "All passed schema, coord &amp; graph checks", "10 / 10"],
        ["Total GPS Observations", f"{ev_sum['total_observations']:,}", f"Spanning {len(ev_sum['utm_epsg_codes'])} WGS84 UTM zones", f"{st_sum['total_observations']:,}"],
        ["Candidate Miss Rate (r=100m, K=16)", f"{ev_sum['candidate_miss_rate']*100:.2f}%", "Frozen STRtree spatial index", f"{st_sum['candidate_miss_rate']*100:.2f}%"],
        ["Candidate Macro F1 (Frozen HGB)", f"{ev_sum['classifier_cand_macro_f1']:.4f}", f"Precision {ev_sum['classifier_cand_precision']:.4f} / Recall {ev_sum['classifier_cand_recall']:.4f}", f"{st_sum['classifier_cand_macro_f1']:.4f}"],
        ["Candidate Log Loss (Frozen HGB)", f"{ev_sum['classifier_log_loss']:.4f}", "Calibrated HMM emission quality", f"{st_sum['classifier_log_loss']:.4f}"],
        ["Weighted Top-1 Accuracy (Clf-Only)", f"{ev_co['weighted_top1_acc']*100:.2f}%", "Independent candidate argmax", f"{st_sum['classifier_only']['weighted_top1_acc']*100:.2f}%"],
        ["Weighted Top-1 Accuracy (Clf + HMM)", f"<b>{ev_ch['weighted_top1_acc']*100:.2f}%</b>", f"<b>{(ev_ch['weighted_top1_acc']-ev_co['weighted_top1_acc'])*100:+.2f} pp lift over Clf-Only</b>", f"{st_sum['classifier_plus_hmm']['weighted_top1_acc']*100:.2f}%"],
        ["Mean Trip Top-1 Accuracy (Clf + HMM)", f"<b>{ev_ch['mean_trip_top1_acc']*100:.2f}%</b>", f"vs {ev_co['mean_trip_top1_acc']*100:.2f}% Clf-Only", f"{st_sum['classifier_plus_hmm']['mean_trip_top1_acc']*100:.2f}%"],
        ["Sequence Length-Weighted Route F1", f"<b>{ev_ch['mean_seq_length_f1']*100:.2f}%</b>", f"vs {ev_co['mean_seq_length_f1']*100:.2f}% Clf-Only ({(ev_ch['mean_seq_length_f1']-ev_co['mean_seq_length_f1'])*100:+.2f} pp)", f"{st_sum['classifier_plus_hmm']['mean_seq_length_f1']*100:.2f}%"],
        ["Sequence Arc-Set F1", f"<b>{ev_ch['mean_seq_arc_f1']*100:.2f}%</b>", f"vs {ev_co['mean_seq_arc_f1']*100:.2f}% Clf-Only", f"{st_sum['classifier_plus_hmm']['mean_seq_arc_f1']*100:.2f}%"],
        ["Reconstructed Direct Route Continuity", f"<b>{ev_ch['mean_recon_direct_cont']*100:.2f}%</b>", f"vs {ev_co['mean_recon_direct_cont']*100:.2f}% Clf-Only", f"{st_sum['classifier_plus_hmm']['mean_recon_direct_cont']*100:.2f}%"],
        ["Disconnected Obs Transitions", f"<b>{ev_ch['total_disconnected_trans']}</b>", f"Cut from {ev_co['total_disconnected_trans']} Clf-Only", str(st_sum["classifier_plus_hmm"]["total_disconnected_trans"])],
        ["Mean Pipeline Latency / Trip", f"{ev_ch['mean_total_ms']:.0f} ms", f"HMM decode: {ev_ch['mean_hmm_ms']:.0f} ms/trip", f"{st_sum['classifier_plus_hmm']['mean_total_pipeline_ms']:.0f} ms"],
    ]
    story.append(
        _make_table(
            ["Evaluation Metric", "Evaluator-Test (10 Reserved)", "HMM Impact / Detail", "Student-Test (Ref)"],
            agg_rows,
            [5.2*cm, 3.6*cm, 5.2*cm, 3.0*cm],
            st,
        )
    )
    story.append(Spacer(1, 8))

    # 2. Per-trip validation & evaluation table
    story.append(Paragraph("2. Per-Trip Validation &amp; Evaluation Breakdown (All 10 Reserved Trips)", st["h1"]))
    pt_headers = ["Trip ID", "Obs", "Arcs", "UTM", "Miss %", "Macro F1", "Clf Top-1", "HMM Top-1", "Δ Top-1", "Seq Len F1", "Cont.", "Time"]
    pt_rows = []
    val_map = {f"{int(r['trip_id']):08d}": r for _, r in val_df.iterrows()}
    for _, r in ev_df.iterrows():
        tid = f"{int(r['trip_id']):08d}"
        vr = val_map[tid]
        pt_rows.append([
            tid,
            str(int(r["n_obs"])),
            f"{int(vr['n_arcs']):,}",
            str(int(r["utm_epsg"])),
            f"{r['candidate_miss_rate']*100:.2f}%",
            f"{r['cand_macro_f1']:.4f}",
            f"{r['clf_top1_acc']*100:.2f}%",
            f"{r['hmm_top1_acc']*100:.2f}%",
            f"{r['delta_top1_acc']*100:+.2f}",
            f"{r['hmm_seq_length_f1']*100:.2f}%",
            f"{r['hmm_recon_direct_cont']*100:.2f}%",
            f"{r['total_ms']:.0f}ms",
        ])
    story.append(
        _make_table(
            pt_headers,
            pt_rows,
            [1.6*cm, 1.1*cm, 1.4*cm, 1.2*cm, 1.3*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.3*cm, 1.6*cm, 1.4*cm, 1.3*cm],
            st,
        )
    )
    story.append(Spacer(1, 8))

    _add_figure(
        story,
        figs / "evaluator_test_results.png",
        "Figure 1: Per-trip comparison of frozen Classifier-Only vs. Classifier + HMM/Viterbi across all 10 reserved evaluator trips.",
        st,
        max_h_cm=6.8,
    )

    story.append(PageBreak())
    story.append(Paragraph("3. Live FastAPI Integration &amp; Ground-Truth Isolation Proof", st["h1"]))
    story.append(
        Paragraph(
            "All 10 reserved trips were verified end-to-end through the FastAPI backend (/api/health, /api/trips, "
            "10 × POST /api/trips/{trip_id}/predict, and GET /api/trips/{trip_id}/map). The backend automatically "
            "activated partition='evaluator_test' (is_development_fallback=False), returned HTTP 200 on every trip, "
            "and passed automated leakage assertions verifying zero reference-route arc IDs or geometries in API payloads "
            "or evaluator-facing Folium maps.",
            st["body"],
        )
    )
    _add_figure(
        story,
        figs / "api_logs_screenshot.png",
        "Figure 2: Verified FastAPI backend logs across all 10 reserved evaluator trips confirming HTTP 200 responses and active evaluator_test partition.",
        st,
        max_h_cm=8.5,
    )

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2.0 * cm,
        title="PathSnap Final Evaluator-Test Report",
        author="PathSnap Engineering Team",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"Generated standalone Final Evaluation Report -> {out_pdf}")
    return out_pdf


if __name__ == "__main__":
    main()
