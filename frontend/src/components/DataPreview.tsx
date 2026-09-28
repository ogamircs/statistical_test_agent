import type { DataPreview as Preview } from "../lib/types";

function cell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
  return String(value);
}

export function DataPreview({ preview, filename }: { preview: Preview; filename: string }) {
  const withMissing = preview.columns.filter((column) => column.missing_pct > 0);
  return (
    <details className="preview">
      <summary>
        Preview <strong>{filename}</strong> · {preview.row_count.toLocaleString()} rows ×{" "}
        {preview.columns.length} columns
        {withMissing.length > 0 && (
          <span className="badge warn">{withMissing.length} with missing values</span>
        )}
      </summary>
      <div className="table-scroll" tabIndex={0}>
        <table>
          <thead>
            <tr>
              {preview.columns.map((column) => (
                <th key={column.name} scope="col">
                  {column.name}
                  <div className="col-meta">
                    {column.dtype}
                    {column.missing_pct > 0 && ` · ${column.missing_pct}% missing`}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, index) => (
              <tr key={index}>
                {preview.columns.map((column) => (
                  <td key={column.name}>{cell(row[column.name])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
