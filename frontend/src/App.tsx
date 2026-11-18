import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Cpu,
  ExternalLink,
  GitBranch,
  MapPin,
  Navigation,
  RefreshCw,
  Route,
  Server,
} from "lucide-react";
import { API_BASE_URL, fetchHealth, fetchTrips, runPrediction } from "./api";
import { RouteMap } from "./components/RouteMap";
import type { HealthInfo, PredictionResponse, TripSummary } from "./types";

function fmtPct(val: number | null | undefined): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "Unavailable";
  return `${(val * 100).toFixed(2)}%`;
}

function fmtNum(val: number | null | undefined, digits = 4): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "Unavailable";
  return val.toFixed(digits);
}

function fmtDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export default function App() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [trips, setTrips] = useState<TripSummary[]>([]);
  const [partitionNotice, setPartitionNotice] = useState<string>("");
  const [isDevFallback, setIsDevFallback] = useState<boolean>(false);

  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [bootError, setBootError] = useState<string | null>(null);

  const [selectedTripId, setSelectedTripId] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [showAllArcs, setShowAllArcs] = useState<boolean>(false);

  const loadBackend = useCallback(async () => {
    setBootError(null);
    try {
      const [h, t] = await Promise.all([fetchHealth(), fetchTrips()]);
      setHealth(h);
      setTrips(t.trips);
      setPartitionNotice(t.partition_notice);
      setIsDevFallback(t.is_development_fallback);
      setBackendOnline(true);
    } catch (err) {
      setBackendOnline(false);
      setBootError(err instanceof Error ? err.message : "Unable to connect to PathSnap API");
    }
  }, []);

  useEffect(() => {
    loadBackend();
  }, [loadBackend]);

  const handleSelectTrip = async (tripId: string) => {
    setSelectedTripId(tripId);
    setPredicting(true);
    setPredictError(null);
    setShowAllArcs(false);
    try {
      const res = await runPrediction(tripId);
      setPrediction(res);
    } catch (err) {
      setPredictError(err instanceof Error ? err.message : "Prediction request failed");
    } finally {
      setPredicting(false);
    }
  };

  const arcList = prediction?.prediction.predicted_route_arc_ids ?? [];
  const displayedArcs = showAllArcs ? arcList : arcList.slice(0, 60);

  return (
    <div className="app-shell">
      {/* Top Header */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <Navigation size={20} />
          </div>
          <div>
            <h1>PathSnap</h1>
            <p className="brand-sub">Distributed Map-Matching &amp; Route Reconstruction</p>
          </div>
        </div>

        <div className="status-strip">
          <div className={`status-pill ${backendOnline ? "pill-ok" : "pill-err"}`}>
            <Server size={14} />
            <span>{backendOnline ? "Backend Connected" : "Backend Offline"}</span>
            <code className="endpoint-code">{API_BASE_URL}</code>
          </div>

          {health && (
            <>
              <div className="status-pill pill-neutral">
                <Cpu size={14} />
                <span>{health.model_name}</span>
              </div>
              <div className="status-pill pill-neutral">
                <GitBranch size={14} />
                <span>HMM/Viterbi: {health.hmm_enabled ? "Active" : "Off"}</span>
              </div>
              <div className="status-pill pill-neutral">
                <MapPin size={14} />
                <span>{health.available_trip_count} Trips</span>
              </div>
            </>
          )}

          <button className="btn-ghost" onClick={loadBackend} title="Refresh backend connection">
            <RefreshCw size={15} />
          </button>
        </div>
      </header>

      {/* Partition Transparency Banner */}
      {backendOnline && (
        <div className={`notice-bar ${isDevFallback ? "notice-dev" : "notice-eval"}`}>
          {isDevFallback ? <AlertTriangle size={15} /> : <CheckCircle2 size={15} />}
          <span>
            <strong>Active Partition:</strong> {health?.active_partition} — {partitionNotice}
          </span>
        </div>
      )}

      {bootError && (
        <div className="error-banner">
          <AlertTriangle size={16} />
          <span>
            Cannot reach FastAPI backend at <code>{API_BASE_URL}</code> ({bootError}). Ensure Laptop 1 is running
            Uvicorn on <code>0.0.0.0:8000</code> and <code>VITE_API_BASE_URL</code> is configured properly.
          </span>
        </div>
      )}

      {/* Main Two-Column Evaluator Workspace */}
      <main className="workspace">
        {/* Left Sidebar: Dynamic Trip Selector */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <h2>
              <Route size={16} />
              Evaluation Trips ({trips.length})
            </h2>
            {isDevFallback && <span className="dev-badge">DEV ONLY</span>}
          </div>

          <div className="trip-list">
            {trips.length === 0 && backendOnline && (
              <div className="empty-state">No trips discovered in configured backend directory.</div>
            )}
            {trips.map((t) => {
              const isSelected = selectedTripId === t.trip_id;
              return (
                <button
                  key={t.trip_id}
                  className={`trip-item ${isSelected ? "trip-item-active" : ""}`}
                  onClick={() => handleSelectTrip(t.trip_id)}
                  disabled={predicting}
                >
                  <div className="trip-item-top">
                    <span className="trip-id">Trip {t.trip_id}</span>
                    <span className="trip-obs">{t.n_gps_observations.toLocaleString()} pts</span>
                  </div>
                  <div className="trip-item-meta">
                    <span>{t.n_road_arcs.toLocaleString()} arcs</span>
                    <span>•</span>
                    <span>{fmtDuration(t.duration_s)}</span>
                    <span>•</span>
                    <span>{t.has_server_side_reference ? "Eval Ready" : "No Ref"}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        {/* Right Workspace: Map + Results Panel */}
        <section className="content">
          {predictError && (
            <div className="error-banner">
              <AlertTriangle size={16} />
              <span>Prediction failed for Trip {selectedTripId}: {predictError}</span>
            </div>
          )}

          <RouteMap prediction={prediction} loading={predicting} />

          {/* Results & Evaluation Metrics Panel */}
          <div className="results-grid">
            {/* KPI Summary Cards */}
            <div className="panel">
              <div className="panel-header">
                <h3>
                  <Activity size={16} />
                  Evaluation &amp; Route Quality Metrics
                </h3>
                {prediction && (
                  <a
                    className="btn-link"
                    href={`${API_BASE_URL}${prediction.map_url}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open Full Folium Map <ExternalLink size={13} />
                  </a>
                )}
              </div>

              {!prediction ? (
                <p className="placeholder-text">
                  Run a prediction to inspect observation-level Top-1 accuracy, Macro F1, route continuity,
                  candidate miss rate, and inference latency.
                </p>
              ) : (
                <>
                  <div className="kpi-grid">
                    <div className="kpi-card kpi-primary">
                      <div className="kpi-label">Top-1 Accuracy (HMM)</div>
                      <div className="kpi-value">{fmtPct(prediction.evaluation.top1_accuracy)}</div>
                      <div className="kpi-sub">
                        Clf-Only: {fmtPct(prediction.evaluation.classifier_only_top1_accuracy)}
                      </div>
                    </div>

                    <div className="kpi-card">
                      <div className="kpi-label">Candidate Macro F1</div>
                      <div className="kpi-value">{fmtNum(prediction.evaluation.macro_f1, 4)}</div>
                      <div className="kpi-sub">Log Loss: {fmtNum(prediction.evaluation.log_loss, 4)}</div>
                    </div>

                    <div className="kpi-card">
                      <div className="kpi-label">Route Continuity</div>
                      <div className="kpi-value">
                        {fmtPct(prediction.continuity.reconstructed_route_direct_continuity)}
                      </div>
                      <div className="kpi-sub">
                        Obs Reachable: {fmtPct(prediction.continuity.network_reachable_continuity)}
                      </div>
                    </div>

                    <div className="kpi-card">
                      <div className="kpi-label">Sequence Route F1</div>
                      <div className="kpi-value">{fmtPct(prediction.evaluation.seq_length_f1)}</div>
                      <div className="kpi-sub">Arc Set F1: {fmtPct(prediction.evaluation.seq_arc_f1)}</div>
                    </div>

                    <div className="kpi-card">
                      <div className="kpi-label">Candidate Miss Rate</div>
                      <div className="kpi-value">{fmtPct(prediction.evaluation.candidate_miss_rate)}</div>
                      <div className="kpi-sub">
                        Hit Top-1: {fmtPct(prediction.evaluation.top1_accuracy_given_hit)}
                      </div>
                    </div>

                    <div className="kpi-card">
                      <div className="kpi-label">Total Inference Time</div>
                      <div className="kpi-value">{prediction.timings_ms.total_ms.toFixed(0)} ms</div>
                      <div className="kpi-sub">
                        Clf {prediction.timings_ms.classifier_ms.toFixed(0)}ms · HMM{" "}
                        {prediction.timings_ms.hmm_viterbi_ms.toFixed(0)}ms
                      </div>
                    </div>
                  </div>

                  {!prediction.evaluation.ground_truth_available && (
                    <div className="info-note">
                      Server-side reference route is unavailable for this trip — supervised metrics are reported
                      as <strong>Unavailable</strong> (distinct from 0%).
                    </div>
                  )}

                  <div className="meta-rows">
                    <div>
                      <span className="meta-k">Matched Observations</span>
                      <span className="meta-v">
                        {prediction.prediction.matched_observation_count.toLocaleString()} /{" "}
                        {prediction.prediction.observation_count.toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="meta-k">HMM/Viterbi Status</span>
                      <span className="meta-v">
                        Active ({prediction.hmm_status.bridged_intermediate_arcs} bridged arcs,{" "}
                        {prediction.hmm_status.segment_breaks} segment breaks,{" "}
                        {prediction.hmm_status.gps_gap_breaks} GPS gaps)
                      </span>
                    </div>
                    <div>
                      <span className="meta-k">Pipeline Latency Breakdown</span>
                      <span className="meta-v">
                        <Clock size={12} style={{ verticalAlign: "-1px", marginRight: 4 }} />
                        Prep {prediction.timings_ms.preprocess_ms.toFixed(0)}ms · Features{" "}
                        {prediction.timings_ms.candidate_and_features_ms.toFixed(0)}ms · Classifier{" "}
                        {prediction.timings_ms.classifier_ms.toFixed(0)}ms · HMM{" "}
                        {prediction.timings_ms.hmm_viterbi_ms.toFixed(0)}ms
                      </span>
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Predicted Road-Segment Sequence Panel */}
            <div className="panel">
              <div className="panel-header">
                <h3>
                  <Route size={16} />
                  Predicted Road-Segment Sequence
                </h3>
                {prediction && (
                  <span className="count-badge">
                    {prediction.prediction.predicted_route_arc_count} directed arcs
                  </span>
                )}
              </div>

              {!prediction ? (
                <p className="placeholder-text">
                  The reconstructed chronological sequence of directed road arc IDs will appear here after
                  decoding.
                </p>
              ) : (
                <>
                  <div className="arc-sequence-box">
                    {displayedArcs.map((aid, i) => (
                      <span key={`${aid}-${i}`} className="arc-chip">
                        {aid}
                      </span>
                    ))}
                    {!showAllArcs && arcList.length > 60 && (
                      <button className="arc-more-btn" onClick={() => setShowAllArcs(true)}>
                        +{arcList.length - 60} more arcs
                      </button>
                    )}
                  </div>
                  {showAllArcs && arcList.length > 60 && (
                    <button className="btn-link-muted" onClick={() => setShowAllArcs(false)}>
                      Collapse arc list
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
