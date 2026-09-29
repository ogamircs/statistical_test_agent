import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { formatBytes } from "../lib/upload";

interface Props {
  disabled: boolean;
  attached: File | null;
  onAttach: (file: File) => void;
  onDetach: () => void;
  /** Resolve to false when the message was not sent; the draft is restored. */
  onSend: (text: string) => void | Promise<boolean | void>;
}

export function Composer({ disabled, attached, onAttach, onDetach, onSend }: Props) {
  const [text, setText] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const textInput = useRef<HTMLTextAreaElement>(null);

  // After picking or dropping a CSV, the natural next step is typing what to
  // do with it; without this, focus stayed on the file button / drop target.
  useEffect(() => {
    if (attached) textInput.current?.focus();
  }, [attached]);

  const send = () => {
    if (disabled || (!text.trim() && !attached)) return;
    const draft = text;
    setText("");
    void Promise.resolve(onSend(draft)).then((sent) => {
      // Rejected (e.g. SESSION_BUSY): give the user their text back unless
      // they already started typing something new.
      if (sent === false) setText((current) => current || draft);
    });
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
          ref={textInput}
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
