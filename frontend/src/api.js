const API_BASE = import.meta.env.VITE_API_URL || "";

async function parseApiError(res, fallback) {
  const err = await res.json().catch(() => ({}));
  const { detail } = err;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  return fallback;
}

export async function analyzeCsv(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/analyze-csv`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(await parseApiError(res, "Failed to analyze CSV"));
  }
  return res.json();
}

export async function processAndTrain(file, targetColumn, imputationStrategy) {
  const form = new FormData();
  form.append("file", file);
  form.append("target_column", targetColumn);
  form.append("imputation_strategy", imputationStrategy);
  const res = await fetch(`${API_BASE}/api/process-and-train`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error(await parseApiError(res, "Training failed"));
  }
  return res.json();
}

export function modelDownloadUrl(modelId) {
  return `${API_BASE}/api/download-model/${modelId}`;
}

export async function downloadCleanedCsv(file, targetColumn, imputationStrategy) {
  const form = new FormData();
  form.append("file", file);
  form.append("target_column", targetColumn);
  form.append("imputation_strategy", imputationStrategy);
  const res = await fetch(`${API_BASE}/api/download-cleaned-csv`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error(await parseApiError(res, "Failed to export cleaned CSV"));
  }
  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match?.[1] || "cleaned.csv";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
