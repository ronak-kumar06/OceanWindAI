"use client";

import type { HistoryItem, TaskStatus } from "@/lib/api";

interface AnalysisPanelProps {
  date: string;
  onDateChange: (date: string) => void;
  hasBounds: boolean;
  loading: boolean;
  status: TaskStatus | null;
  history: HistoryItem[];
  onAnalyze: () => void;
  onSelectHistory: (item: HistoryItem) => void;
}

export default function AnalysisPanel({
  date,
  onDateChange,
  hasBounds,
  loading,
  status,
  history,
  onAnalyze,
  onSelectHistory,
}: AnalysisPanelProps) {
  return (
    <aside className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div>
        <h1 className="text-xl font-semibold text-sky-100">OceanWind AI</h1>
        <p className="mt-1 text-sm text-slate-400">
          Click the map to select a coastal region, then run SAR wind analysis.
        </p>
      </div>

      <div className="space-y-2">
        <label className="text-xs font-medium uppercase tracking-wide text-slate-400">
          Acquisition date
        </label>
        <input
          type="date"
          value={date}
          onChange={(e) => onDateChange(e.target.value)}
          className="w-full rounded-md border border-slate-700 bg-ocean-900 px-3 py-2 text-sm text-white"
        />
      </div>

      <button
        onClick={onAnalyze}
        disabled={!hasBounds || loading}
        className="rounded-md bg-sky-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? "Analyzing…" : "Run Analysis"}
      </button>

      {status && (
        <div className="rounded-md border border-slate-700 bg-ocean-900 p-3 text-sm">
          <p className="font-medium capitalize text-sky-200">{status.status}</p>
          {status.stats?.mean_speed != null && (
            <ul className="mt-2 space-y-1 text-slate-300">
              <li>Mean speed: {status.stats.mean_speed.toFixed(2)} m/s</li>
              <li>Dominant dir: {status.stats.dominant_dir?.toFixed(1)}°</li>
              <li>Source: {status.stats.data_source}</li>
            </ul>
          )}
          {status.image_url && (
            <img
              src={status.image_url}
              alt="Wind field quiver plot"
              className="mt-3 w-full rounded border border-slate-700"
            />
          )}
        </div>
      )}

      <div>
        <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
          Recent analyses
        </h2>
        <ul className="space-y-2">
          {history.slice(0, 8).map((item) => (
            <li key={item.id}>
              <button
                onClick={() => onSelectHistory(item)}
                className="w-full rounded-md border border-slate-800 bg-ocean-950 px-3 py-2 text-left text-sm hover:border-sky-700"
              >
                <span className="text-sky-200">#{item.id}</span>{" "}
                <span className="text-slate-400">{item.date_selected}</span>
                <span className="ml-2 capitalize text-slate-500">{item.status}</span>
              </button>
            </li>
          ))}
          {history.length === 0 && (
            <p className="text-sm text-slate-500">No analyses yet.</p>
          )}
        </ul>
      </div>
    </aside>
  );
}
