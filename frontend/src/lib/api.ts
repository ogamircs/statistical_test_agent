import { SseParser } from "./sse";
import type {
  ChartSpec,
  ChatMessage,
  PublicConfig,
  SessionSummary,
  StreamEvent,
  UploadResult,
} from "./types";

const TOKEN_KEY = "statagent.token";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

function readToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function storeToken(token: string | null): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage unavailable (private mode); the token lives for this page only.
  }
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const token = readToken();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: headers((init.headers as Record<string, string>) ?? {}),
  });
  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function toApiError(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as { error?: { code?: string; message?: string } };
    return new ApiError(
      response.status,
      body.error?.code ?? "HTTP_ERROR",
      body.error?.message ?? response.statusText,
    );
  } catch {
    return new ApiError(response.status, "HTTP_ERROR", response.statusText || "Request failed");
  }
}

export const api = {
  config: () => request<PublicConfig>("/api/config"),

  login: async (username: string, password: string) => {
    const { token } = await request<{ token: string }>("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    storeToken(token);
  },

  listSessions: async () => (await request<{ sessions: SessionSummary[] }>("/api/sessions")).sessions,

  createSession: async () => (await request<{ id: string }>("/api/sessions", { method: "POST" })).id,

  deleteSession: (id: string) => request<void>(`/api/sessions/${id}`, { method: "DELETE" }),

  messages: (id: string) =>
    request<{ messages: Omit<ChatMessage, "id">[]; charts: ChartSpec[] }>(`/api/sessions/${id}/messages`),

  clearMessages: (id: string) => request<void>(`/api/sessions/${id}/messages`, { method: "DELETE" }),

  charts: async (id: string, type: string) =>
    (
      await request<{ charts: ChartSpec[] }>(
        `/api/sessions/${id}/charts?type=${encodeURIComponent(type)}`,
      )
    ).charts,

  upload: (id: string, file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<UploadResult>(`/api/sessions/${id}/upload`, { method: "POST", body });
  },

  /** POST a chat message and yield SSE events as they stream in. */
  async *chat(
    id: string,
    message: string,
    fileId: string | null,
    signal?: AbortSignal,
  ): AsyncGenerator<StreamEvent> {
    const response = await fetch(`/api/sessions/${id}/chat`, {
      method: "POST",
      headers: headers({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify({ message, file_id: fileId }),
      signal,
    });
    if (!response.ok || !response.body) throw await toApiError(response);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const parser = new SseParser();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      yield* parser.push(decoder.decode(value, { stream: true }));
    }
    yield* parser.push(decoder.decode() + "\n\n");
  },
};
