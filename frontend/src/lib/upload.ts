export type UploadCheck = { ok: true } | { ok: false; reason: string };

/** Client-side mirror of the server's upload rules (server re-validates). */
export function validateCsvFile(file: Pick<File, "name" | "size">, maxMb: number): UploadCheck {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    return { ok: false, reason: "Only .csv files can be uploaded." };
  }
  if (file.size === 0) {
    return { ok: false, reason: "The file is empty." };
  }
  if (file.size > maxMb * 1024 * 1024) {
    return { ok: false, reason: `File exceeds the ${maxMb} MB upload limit.` };
  }
  return { ok: true };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
