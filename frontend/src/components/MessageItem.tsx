import type { ChatMessage } from "../lib/types";
import { DataPreview } from "./DataPreview";
import { Markdown } from "./Markdown";
import { ProgressSteps } from "./ProgressSteps";

export function MessageItem({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <article className="msg msg-user" aria-label="You">
        {message.attachment && (
          <div className="attachment" title={message.attachment}>
            <span aria-hidden="true">▤</span> {message.attachment}
          </div>
        )}
        {message.content && <p className="user-text">{message.content}</p>}
        {message.preview && message.attachment && (
          <DataPreview preview={message.preview} filename={message.attachment} />
        )}
      </article>
    );
  }
  return (
    <article
      className={`msg msg-assistant${message.errorCode ? " msg-error" : ""}`}
      aria-label="Agent"
      aria-busy={message.pending ? true : undefined}
    >
      <ProgressSteps steps={message.steps ?? []} pending={!!message.pending} />
      {message.content && <Markdown content={message.content} />}
      {message.errorCode && <div className="error-code">Error code: {message.errorCode}</div>}
    </article>
  );
}
