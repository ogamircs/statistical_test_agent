import { useRef, useState, type KeyboardEvent } from "react";
import { formatBytes } from "../lib/upload";

interface Props {
  disabled: boolean;
  attached: File | null;
  onAttach: (file: File) => void;
  onDetach: () => void;
  onSend: (text: string) => void;
}

export function Composer({ disabled, attached, onAttach, onDetach, onSend }: Props) {
  const [text, setText] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  const send = () => {
    if (disabled || (!text.trim() && !attached)) return;
    onSend(text);
    setText("");
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send();
    }
  };

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault();
        send();
      }}
    >
      {attached && (
        <div className="attachment attachment-pending">
          <span aria-hidden="true">▤</span> {attached.name}
          <span className="muted"> · {formatBytes(attached.size)}</span>
          <button type="button" className="icon-btn" onClick={onDetach} aria-label={`Remove ${attached.name}`}>
            ✕
          </button>
        </div>
      )}
      <div className="composer-row">
        <button
          type="button"
          className="icon-btn attach"
          onClick={() => fileInput.current?.click()}
          aria-label="Attach a CSV file"
          title="Attach a CSV file"
          disabled={disabled}
        >
          ＋
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".csv,text/csv"
          hidden
          data-testid="file-input"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onAttach(file);
            event.target.value = "";
          }}
        />
        <label className="visually-hidden" htmlFor="composer-input">
          Message
        </label>
        <textarea
          id="composer-input"
          value={text}
          rows={1}
          placeholder={attached ? "Say what to do with this file, or just send" : "Ask about your experiment…"}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <button type="submit" className="send" disabled={disabled || (!text.trim() && !attached)}>
          Send
        </button>
      </div>
      <p className="hint muted">Enter to send · Shift+Enter for a new line · drop a CSV anywhere</p>
    </form>
  );
}
