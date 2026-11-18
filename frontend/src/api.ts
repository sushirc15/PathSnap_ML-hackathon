import type { HealthInfo, PredictionResponse, TripsResponse } from "./types";

export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore non-JSON body
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${API_BASE_URL}/api/health`);
  return handleResponse<HealthInfo>(res);
}

export async function fetchTrips(): Promise<TripsResponse> {
  const res = await fetch(`${API_BASE_URL}/api/trips`);
  return handleResponse<TripsResponse>(res);
}

export async function runPrediction(tripId: string): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE_URL}/api/trips/${encodeURIComponent(tripId)}/predict`, {
    method: "POST",
  });
  return handleResponse<PredictionResponse>(res);
}
