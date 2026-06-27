const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export interface AnalysisStats {
  mean_speed?: number;
  dominant_dir?: number;
  n_vectors?: number;
  data_source?: string;
}

export interface TaskStatus {
  task_id: number;
  status: "pending" | "processing" | "completed" | "failed";
  image_url?: string;
  stats?: AnalysisStats;
}

export interface HistoryItem {
  id: number;
  date_selected: string;
  bbox_str?: string;
  status: string;
  data_source?: string;
  image_url?: string;
  mean_speed?: number;
  dominant_dir?: number;
  n_vectors?: number;
  created_at?: string;
}

export async function submitAnalysis(date: string, bbox: string): Promise<{ task_id: number }> {
  const res = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date_selected: date, bbox }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getTaskStatus(taskId: number): Promise<TaskStatus> {
  const res = await fetch(`${API_URL}/api/status/${taskId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getHistory(): Promise<HistoryItem[]> {
  const res = await fetch(`${API_URL}/api/history`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function bboxToString(bounds: [[number, number], [number, number]]): string {
  const [[south, west], [north, east]] = bounds;
  return `${west},${south},${east},${north}`;
}

export async function pollUntilComplete(
  taskId: number,
  onUpdate?: (status: TaskStatus) => void,
  intervalMs = 2000,
  maxAttempts = 120,
): Promise<TaskStatus> {
  for (let i = 0; i < maxAttempts; i++) {
    const status = await getTaskStatus(taskId);
    onUpdate?.(status);
    if (status.status === "completed" || status.status === "failed") {
      return status;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("Analysis timed out");
}
