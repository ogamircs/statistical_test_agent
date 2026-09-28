import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Composer } from "./Composer";
import { Markdown } from "./Markdown";
import { MessageItem } from "./MessageItem";

describe("Markdown", () => {
  it("renders agent GFM tables as real, scrollable tables", () => {
    const content = "## Results\n\n| Segment | p-value |\n|---|---|\n| Premium | 0.0010* |\n";
    render(<Markdown content={content} />);

    const table = screen.getByRole("table");
    expect(table.parentElement).toHaveClass("table-scroll");
    expect(within(table).getAllByRole("columnheader").map((th) => th.textContent)).toEqual([
      "Segment",
      "p-value",
    ]);
    expect(within(table).getByRole("cell", { name: "Premium" })).toBeInTheDocument();
  });

  it("does not render raw HTML from agent/data content", () => {
    const { container } = render(<Markdown content={'<img src=x onerror="alert(1)"> text'} />);
    expect(container.querySelector("img")).toBeNull();
  });
});

describe("MessageItem", () => {
  it("shows progress steps and error codes for assistant messages", () => {
    render(
      <MessageItem
        message={{
          id: "1",
          role: "assistant",
          content: "Failed",
          errorCode: "LLM_TIMEOUT",
          steps: [{ id: "s", name: "load_csv", label: "Loading data", status: "done" }],
        }}
      />,
    );
    expect(screen.getByText("Loading data")).toBeInTheDocument();
    expect(screen.getByText("Error code: LLM_TIMEOUT")).toBeInTheDocument();
  });

  it("shows the attachment name and a collapsible data preview for user uploads", () => {
    render(
      <MessageItem
        message={{
          id: "2",
          role: "user",
          content: "best guess",
          attachment: "exp.csv",
          preview: {
            row_count: 1200,
            columns: [
              { name: "group", dtype: "object", missing_pct: 0 },
              { name: "revenue", dtype: "float64", missing_pct: 2.5 },
            ],
            rows: [{ group: "control", revenue: 12.5 }],
          },
        }}
      />,
    );
    expect(screen.getAllByText("exp.csv").length).toBeGreaterThan(0);
    expect(screen.getByText(/1,200 rows × 2 columns/)).toBeInTheDocument();
    expect(screen.getByText("1 with missing values")).toBeInTheDocument();
  });
});

describe("Composer", () => {
  it("sends on Enter, inserts a newline on Shift+Enter", async () => {
    const onSend = vi.fn();
    render(<Composer disabled={false} attached={null} onAttach={vi.fn()} onDetach={vi.fn()} onSend={onSend} />);
    const input = screen.getByLabelText("Message");

    await userEvent.type(input, "line one{Shift>}{Enter}{/Shift}line two");
    expect(onSend).not.toHaveBeenCalled();
    await userEvent.type(input, "{Enter}");

    expect(onSend).toHaveBeenCalledWith("line one\nline two");
    expect(input).toHaveValue("");
  });

  it("allows sending an attachment without text and hands picked files to onAttach", async () => {
    const onSend = vi.fn();
    const onAttach = vi.fn();
    const file = new File(["a,b\n1,2\n"], "exp.csv", { type: "text/csv" });
    const { rerender } = render(
      <Composer disabled={false} attached={null} onAttach={onAttach} onDetach={vi.fn()} onSend={onSend} />,
    );
    await userEvent.upload(screen.getByTestId("file-input"), file);
    expect(onAttach).toHaveBeenCalledWith(file);

    rerender(<Composer disabled={false} attached={file} onAttach={onAttach} onDetach={vi.fn()} onSend={onSend} />);
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledWith("");
  });

  it("disables sending while busy", () => {
    render(<Composer disabled attached={null} onAttach={vi.fn()} onDetach={vi.fn()} onSend={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });
});
