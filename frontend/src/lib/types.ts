import type { Data, Layout } from "plotly.js";

export interface ChartSpec {
  name: string;
  title: string;
  figure: { data: Data[]; layout: Partial<Layout> };
}

export interface ChartTypeOption {
  key: string;
  label: string;
}

export interface PublicConfig {
  auth_required: boolean;
  max_upload_mb: number;
  chart_types: ChartTypeOption[];
  model: string;
}

export interface SessionSummary {
  id: string;
  title: string;
  updated_at: string;
  message_count: number;
}

export interface PreviewColumn {
  name: string;
  dtype: string;
  missing_pct: number;
}

export interface DataPreview {
  row_count: number;
  columns: PreviewColumn[];
  rows: Record<string, unknown>[];
}

export interface UploadResult {
  file_id: string;
  filename: string;
  size_bytes: number;
  preview: DataPreview;
}

export interface ProgressStep {
  id: string;
  name: string;
  label: string;
  status: "running" | "done" | "failed";
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachment?: string | null;
  preview?: DataPreview;
  steps?: ProgressStep[];
  pending?: boolean;
  errorCode?: string | null;
}

export type StreamEvent =
  | { event: "status"; data: { state: string } }
  | { event: "tool_start"; data: { id: string; name: string; label: string } }
  | { event: "tool_end"; data: { id: string; name: string; ok: boolean } }
  | { event: "message"; data: { content: string; error_code: string | null } }
  | { event: "charts"; data: { charts: ChartSpec[] } }
  | { event: "error"; data: { code: string; message: string } }
  | { event: "done"; data: Record<string, never> };
