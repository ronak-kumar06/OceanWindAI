"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useState } from "react";
import type { LatLngBounds } from "leaflet";
import AnalysisPanel from "@/components/AnalysisPanel";
import {
  bboxToString,
  getHistory,
  pollUntilComplete,
  submitAnalysis,
  type HistoryItem,
  type TaskStatus,
} from "@/lib/api";

const WindMap = dynamic(() => import("@/components/WindMap"), { ssr: false });

export default function HomePage() {
  const [date, setDate] = useState("2024-06-15");
  const [bounds, setBounds] = useState<LatLngBounds | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<TaskStatus | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const refreshHistory = useCallback(async () => {
    try {
      const items = await getHistory();
      setHistory(items);
    } catch {
      // API may be offline during local dev
    }
  }, []);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  const handleAnalyze = async () => {
    if (!bounds) return;
    setLoading(true);
    setStatus(null);
    try {
      const bbox = bboxToString([
        [bounds.getSouth(), bounds.getWest()],
        [bounds.getNorth(), bounds.getEast()],
      ]);
      const { task_id } = await submitAnalysis(date, bbox);
      const result = await pollUntilComplete(task_id, setStatus);
      setStatus(result);
      await refreshHistory();
    } catch (err) {
      setStatus({
        task_id: -1,
        status: "failed",
        stats: { data_source: String(err) },
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSelectHistory = (item: HistoryItem) => {
    setStatus({
      task_id: item.id,
      status: item.status as TaskStatus["status"],
      image_url: item.image_url,
      stats: {
        mean_speed: item.mean_speed,
        dominant_dir: item.dominant_dir,
        n_vectors: item.n_vectors,
        data_source: item.data_source,
      },
    });
  };

  return (
    <main className="grid h-screen grid-cols-1 lg:grid-cols-[340px_1fr]">
      <AnalysisPanel
        date={date}
        onDateChange={setDate}
        hasBounds={bounds !== null}
        loading={loading}
        status={status}
        history={history}
        onAnalyze={handleAnalyze}
        onSelectHistory={handleSelectHistory}
      />
      <div className="relative min-h-[50vh] p-4 lg:min-h-0">
        <WindMap bounds={bounds} onBoundsChange={setBounds} />
      </div>
    </main>
  );
}
