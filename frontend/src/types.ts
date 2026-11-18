export interface HealthInfo {
  status: string;
  service: string;
  model_name: string;
  model_trained_on: string;
  n_features: number;
  hmm_enabled: boolean;
  hmm_config: Record<string, number>;
  candidate_config: {
    radius_m: number;
    fallback_radius_m: number;
    top_k: number;
  };
  active_partition: string;
  is_development_fallback: boolean;
  available_trip_count: number;
}

export interface TripSummary {
  trip_id: string;
  partition: string;
  is_development_only: boolean;
  n_gps_observations: number;
  n_road_nodes: number;
  n_road_arcs: number;
  duration_s: number;
  center_lat: number;
  center_lon: number;
  has_server_side_reference: boolean;
}

export interface TripsResponse {
  active_partition: string;
  is_development_fallback: boolean;
  partition_notice: string;
  trip_count: number;
  trips: TripSummary[];
}

export interface GpsPoint {
  idx: number;
  lat: number;
  lon: number;
  t: number;
}

export interface PredictionResponse {
  trip_id: string;
  partition: string;
  is_development_fallback: boolean;
  model_name: string;
  hmm_status: {
    enabled: boolean;
    algorithm: string;
    bridged_intermediate_arcs: number;
    segment_breaks: number;
    gps_gap_breaks: number;
  };
  prediction: {
    observation_count: number;
    matched_observation_count: number;
    predicted_route_arc_count: number;
    predicted_route_arc_ids: number[];
    utm_crs: string;
  };
  continuity: {
    direct_step_continuity: number;
    network_reachable_continuity: number;
    transition_reachable_continuity: number;
    reconstructed_route_direct_continuity: number;
    disconnected_transitions: number;
  };
  evaluation: {
    ground_truth_available: boolean;
    top1_accuracy: number | null;
    top1_accuracy_given_hit: number | null;
    classifier_only_top1_accuracy: number | null;
    candidate_miss_rate: number | null;
    macro_f1: number | null;
    log_loss: number | null;
    seq_arc_f1: number | null;
    seq_length_f1: number | null;
  };
  timings_ms: {
    preprocess_ms: number;
    candidate_and_features_ms: number;
    classifier_ms: number;
    hmm_viterbi_ms: number;
    total_ms: number;
  };
  validation_warnings: string[];
  map_url: string;
  geometry: {
    bounds: [[number, number], [number, number]];
    start: [number, number];
    end: [number, number];
    gps_points: GpsPoint[];
    predicted_polylines: [number, number][][];
    local_road_segments: [number, number][][];
  };
}
