import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { ApiError, api, storeToken } from "./lib/api";
import type { ChartSpec, ChatMessage, PublicConfig, SessionSummary } from "./lib/types";
import { prefersReducedMotion, useTheme } from "./lib/theme";
import { validateCsvFile } from "./lib/upload";
import { ChartWorkspace } from "./components/ChartWorkspace";
import { Composer } from "./components/Composer";
import { Login } from "./components/Login";
import { MessageItem } from "./components/MessageItem";
import { Sidebar } from "./components/Sidebar";

const STARTERS = [
  {
    title: "Try the sample dataset",
    text: "Run a best guess analysis of the sample dataset at data/sample_ab_data.csv",
  },
  {
    title: "Plan a sample size",
    text: "How many users per arm do I need to detect a 2 percentage point lift on a 10% conversion rate at 80% power?",
  },
  {
    title: "What can you do?",
    text: "What kinds of A/B test analyses can you run, and what should my CSV look like?",
  },
];

let nextId = 0;
const newId = () => `m${Date.now()}-${nextId++}`;

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [needsLogin, setNeedsLogin] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [charts, setCharts] = useState<ChartSpec[]>([]);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [attached, setAttached] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);

  const handleError = useCallback((error: unknown) => {
    if (error instanceof ApiError && error.status === 401) {
      storeToken(null);
      setNeedsLogin(true);
      return;
    }
    setNotice(error instanceof Error ? error.message : "Something went wrong.");
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await api.listSessions());
    } catch (error) {
      handleError(error);
    }
  }, [handleError]);

  useEffect(() => {
    api
      .config()
      .then((loaded) => {
        setConfig(loaded);
        return refreshSessions();
      })
      .catch(handleError);
  }, [refreshSessions, handleError]);

  useEffect(() => {
    scroller.current?.scrollTo({
      top: scroller.current.scrollHeight,
      behavior: prefersReducedMotion() ? "auto" : "smooth",
    });
  }, [messages]);

  const updateAssistant = (id: string, patch: (message: ChatMessage) => ChatMessage) =>
    setMessages((current) => current.map((message) => (message.id === id ? patch(message) : message)));

  const attach = (file: File) => {
    const check = validateCsvFile(file, config?.max_upload_mb ?? 50);
    if (!check.ok) {
      setNotice(check.reason);
      return;
    }
    setNotice(null);
    setAttached(file);
  };

  const send = async (text: string) => {
    if (busy) return;
    setBusy(true);
    setNotice(null);
    const file = attached;
    try {
      const sessionId = activeId ?? (await api.createSession());
      if (!activeId) setActiveId(sessionId);

      let fileId: string | null = null;
      const userMessage: ChatMessage = { id: newId(), role: "user", content: text.trim() };
      if (file) {
        const uploaded = await api.upload(sessionId, file);
        fileId = uploaded.file_id;
        userMessage.attachment = uploaded.filename;
        userMessage.preview = uploaded.preview;
        setAttached(null);
      }
      const assistantId = newId();
      setMessages((current) => [
        ...current,
        userMessage,
        { id: assistantId, role: "assistant", content: "", steps: [], pending: true },
      ]);

      try {
        for await (const event of api.chat(sessionId, text, fileId)) {
          switch (event.event) {
            case "tool_start":
              updateAssistant(assistantId, (m) => ({
                ...m,
                steps: [
                  ...(m.steps ?? []),
                  { id: event.data.id, name: event.data.name, label: event.data.label, status: "running" },
                ],
              }));
              break;
            case "tool_end":
              updateAssistant(assistantId, (m) => ({
                ...m,
                steps: (m.steps ?? []).map((step) =>
                  step.id === event.data.id ? { ...step, status: event.data.ok ? "done" : "failed" } : step,
                ),
              }));
              break;
            case "message":
              updateAssistant(assistantId, (m) => ({
                ...m,
                content: event.data.content,
                errorCode: event.data.error_code,
              }));
              break;
            case "charts":
              if (event.data.charts.length > 0) {
                setCharts(event.data.charts);
                setChartError(null);
                setWorkspaceOpen(true);
              }
              break;
            case "error":
              updateAssistant(assistantId, (m) => ({
                ...m,
                content: event.data.message,
                errorCode: event.data.code,
              }));
              break;
            default:
              break;
          }
        }
      } finally {
        updateAssistant(assistantId, (m) => ({
          ...m,
          pending: false,
          content: m.content || (m.errorCode ? m.content : "_The connection closed before a reply arrived._"),
        }));
      }
      await refreshSessions();
    } catch (error) {
      handleError(error);
    } finally {
      setBusy(false);
    }
  };

  const openSession = async (id: string) => {
    if (busy) return;
    try {
      const history = await api.messages(id);
      setActiveId(id);
      setMessages(history.messages.map((message) => ({ ...message, id: newId() })));
      setCharts(history.charts);
      setChartError(null);
      setAttached(null);
      setWorkspaceOpen(history.charts.length > 0);
    } catch (error) {
      handleError(error);
    }
  };

  const newSession = () => {
    if (busy) return;
    setActiveId(null);
    setMessages([]);
    setCharts([]);
    setAttached(null);
    setWorkspaceOpen(false);
  };

  const deleteSession = async (id: string) => {
    try {
      await api.deleteSession(id);
      if (id === activeId) newSession();
      await refreshSessions();
    } catch (error) {
      handleError(error);
    }
  };

  const clearSession = async () => {
    if (!activeId || busy) return;
    try {
      await api.clearMessages(activeId);
      setMessages([]);
      setCharts([]);
      await refreshSessions();
    } catch (error) {
      handleError(error);
    }
  };

  const requestCharts = async (type: string) => {
    if (!activeId) return;
    setChartLoading(true);
    setChartError(null);
    try {
      setCharts(await api.charts(activeId, type));
    } catch (error) {
      if (error instanceof ApiError && error.status !== 401) setChartError(error.message);
      else handleError(error);
    } finally {
      setChartLoading(false);
    }
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) attach(file);
  };

  if (needsLogin) {
    return (
      <Login
        onSuccess={() => {
          setNeedsLogin(false);
          void refreshSessions();
        }}
      />
    );
  }

  return (
    <div className={`app${workspaceOpen ? " with-workspace" : ""}`}>
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        theme={theme}
        onNew={newSession}
        onSelect={(id) => void openSession(id)}
        onDelete={(id) => void deleteSession(id)}
        onClear={() => void clearSession()}
        onToggleTheme={toggleTheme}
      />

      <main
        className={`chat${dragging ? " dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(event) => {
          if (event.currentTarget === event.target) setDragging(false);
        }}
        onDrop={onDrop}
      >
        <header className="chat-header">
          <h1>{sessions.find((s) => s.id === activeId)?.title ?? "New analysis"}</h1>
          <button
            type="button"
            className="charts-toggle"
            onClick={() => setWorkspaceOpen(!workspaceOpen)}
            aria-expanded={workspaceOpen}
          >
            📊 Charts{charts.length > 0 ? ` (${charts.length})` : ""}
          </button>
        </header>

        <div className="chat-scroll" ref={scroller}>
          {messages.length === 0 ? (
            <section className="empty-state">
              <h2>Analyze an A/B test</h2>
              <p className="muted">
                Upload a CSV with a group column and an outcome metric — or start from one of these.
              </p>
              <div className="starters">
                {STARTERS.map((starter) => (
                  <button
                    key={starter.title}
                    type="button"
                    className="starter"
                    disabled={busy}
                    onClick={() => void send(starter.text)}
                  >
                    <strong>{starter.title}</strong>
                    <span>{starter.text}</span>
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <div className="messages" role="log" aria-live="polite" aria-relevant="additions">
              {messages.map((message) => (
                <MessageItem key={message.id} message={message} />
              ))}
            </div>
          )}
        </div>

        {notice && (
          <div className="notice" role="alert">
            {notice}
            <button type="button" className="icon-btn" onClick={() => setNotice(null)} aria-label="Dismiss">
              ✕
            </button>
          </div>
        )}

        <Composer
          disabled={busy}
          attached={attached}
          onAttach={attach}
          onDetach={() => setAttached(null)}
          onSend={(text) => void send(text)}
        />
        {dragging && (
          <div className="drop-overlay" aria-hidden="true">
            Drop a CSV to attach it
          </div>
        )}
      </main>

      {workspaceOpen && (
        <ChartWorkspace
          charts={charts}
          chartTypes={config?.chart_types ?? [{ key: "dashboard", label: "Dashboard" }]}
          theme={theme}
          loading={chartLoading}
          error={chartError}
          canRequest={!!activeId && messages.length > 0 && !busy}
          onRequest={(type) => void requestCharts(type)}
          onClose={() => setWorkspaceOpen(false)}
        />
      )}
    </div>
  );
}
