import type { SessionSummary } from "../lib/types";
import type { Theme } from "../lib/theme";

interface Props {
  sessions: SessionSummary[];
  activeId: string | null;
  theme: Theme;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onClear: () => void;
  onToggleTheme: () => void;
}

function relativeTime(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return new Date(iso).toLocaleDateString();
}

export function Sidebar({
  sessions,
  activeId,
  theme,
  onNew,
  onSelect,
  onDelete,
  onClear,
  onToggleTheme,
}: Props) {
  return (
    <nav className="sidebar" aria-label="Conversations">
      <div className="brand">
        <img src="/favicon.svg" alt="" width={24} height={24} />
        <span>A/B Testing Agent</span>
      </div>
      <button type="button" className="new-chat" onClick={onNew}>
        ＋ New analysis
      </button>
      <h2 className="sidebar-heading">History</h2>
      {sessions.length === 0 ? (
        <p className="muted small">Your conversations will appear here.</p>
      ) : (
        <ul className="session-list">
          {sessions.map((session) => (
            <li key={session.id} className={session.id === activeId ? "active" : undefined}>
              <button
                type="button"
                className="session-open"
                onClick={() => onSelect(session.id)}
                aria-current={session.id === activeId ? "page" : undefined}
              >
                <span className="session-title">{session.title}</span>
                <span className="session-time">{relativeTime(session.updated_at)}</span>
              </button>
              <button
                type="button"
                className="icon-btn session-delete"
                onClick={() => onDelete(session.id)}
                aria-label={`Delete conversation: ${session.title}`}
                title="Delete conversation"
              >
                🗑
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="sidebar-footer">
        <button type="button" onClick={onClear} disabled={!activeId}>
          Clear this conversation
        </button>
        <button type="button" onClick={onToggleTheme} aria-label="Toggle dark mode">
          {theme === "dark" ? "☀ Light" : "☾ Dark"}
        </button>
      </div>
    </nav>
  );
}
