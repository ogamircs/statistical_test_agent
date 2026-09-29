import type { ProgressStep } from "../lib/types";

const ICON: Record<ProgressStep["status"], string> = {
  running: "…",
  done: "✓",
  failed: "✕",
};

export function ProgressSteps({ steps, pending }: { steps: ProgressStep[]; pending: boolean }) {
  if (steps.length === 0 && !pending) return null;
  const summary = pending
    ? steps.at(-1)?.status === "running"
      ? `${steps.at(-1)?.label}…`
      : "Thinking…"
    : `${steps.length} step${steps.length === 1 ? "" : "s"}`;
  return (
    <details className="steps" open={pending}>
      <summary>
        {pending && <span className="spinner" aria-hidden="true" />}
        <span aria-live="polite">{summary}</span>
      </summary>
      <ol>
        {steps.map((step) => (
          <li key={step.id} className={`step step-${step.status}`}>
            <span className="step-icon" aria-hidden="true">
              {ICON[step.status]}
            </span>
            <span>{step.label}</span>
            <span className="visually-hidden">
              {step.status === "running" ? "in progress" : step.status === "done" ? "done" : "failed"}
            </span>
          </li>
        ))}
      </ol>
    </details>
  );
}
