import { Download, FileSpreadsheet } from "lucide-react";
import { modelDownloadUrl } from "../api.js";

function KpiCard({ label, value }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-gradient-to-br from-white to-slate-50 p-4 shadow-sm dark:border-slate-700 dark:from-slate-900 dark:to-slate-950">
      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-bold text-accent">
        {(value * 100).toFixed(2)}%
      </p>
    </div>
  );
}

function ConfusionMatrix({ matrix }) {
  if (!matrix?.length) return null;
  const flatMax = Math.max(...matrix.flat(), 1);

  return (
    <div className="rounded-xl border border-slate-200 p-4 dark:border-slate-700">
      <h4 className="mb-3 text-sm font-semibold uppercase text-slate-500">Confusion matrix</h4>
      <div
        className="inline-grid gap-1"
        style={{ gridTemplateColumns: `repeat(${matrix[0].length}, minmax(3rem, 1fr))` }}
      >
        {matrix.map((row, i) =>
          row.map((cell, j) => {
            const intensity = cell / flatMax;
            return (
              <div
                key={`${i}-${j}`}
                className="flex aspect-square items-center justify-center rounded-md text-sm font-bold text-white"
                style={{
                  backgroundColor: `rgba(99, 102, 241, ${0.25 + intensity * 0.75})`,
                }}
                title={`Actual ${i}, Predicted ${j}: ${cell}`}
              >
                {cell}
              </div>
            );
          })
        )}
      </div>
      <p className="mt-2 text-xs text-slate-500">Rows: actual · Columns: predicted</p>
    </div>
  );
}

export default function DashboardMetrics({ result, onDownloadCleaned, downloading }) {
  if (!result) return null;

  const { metrics, baseline_scores, winner, model_id, warnings, rows_used, classes_used } =
    result;
  const baselines = baseline_scores || {};

  const handleDownload = async () => {
    const url = modelDownloadUrl(model_id);
    const res = await fetch(url);
    if (!res.ok) throw new Error("Download failed");
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `model_${model_id}.pkl`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <section className="mt-8 space-y-6 rounded-2xl border border-indigo-200 bg-white p-6 shadow-lg dark:border-indigo-900/50 dark:bg-slate-900">
      <div>
        <h2 className="text-2xl font-bold">AutoML results</h2>
        <p className="text-slate-500 dark:text-slate-400">
          Winner after tuning: <span className="font-semibold text-accent">{winner}</span>
        </p>
        {rows_used != null && (
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Trained on {rows_used.toLocaleString()} rows · {classes_used} classes
          </p>
        )}
      </div>

      {warnings?.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
          <p className="font-semibold">Training notes</p>
          <ul className="mt-1 list-inside list-disc">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <h3 className="mb-3 text-sm font-semibold uppercase text-slate-500">
          Baseline accuracy comparison
        </h3>
        <div className="grid gap-3 sm:grid-cols-3">
          {Object.entries(baselines).map(([name, score]) => (
            <div
              key={name}
              className="rounded-xl border border-slate-200 p-4 dark:border-slate-700"
            >
              <p className="text-sm font-medium text-slate-600 dark:text-slate-300">{name}</p>
              <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                <div
                  className="h-full rounded-full bg-accent transition-all"
                  style={{ width: `${Math.min(score * 100, 100)}%` }}
                />
              </div>
              <p className="mt-1 text-lg font-bold">{(score * 100).toFixed(2)}%</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard label="Accuracy" value={metrics.accuracy} />
        <KpiCard label="Precision" value={metrics.precision} />
        <KpiCard label="Recall" value={metrics.recall} />
        <KpiCard label="F1-Score" value={metrics.f1_score} />
      </div>

      <ConfusionMatrix matrix={metrics.confusion_matrix} />

      <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
        <button
          type="button"
          onClick={onDownloadCleaned}
          disabled={downloading}
          className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl border-2 border-accent bg-white px-6 py-4 text-lg font-bold text-accent transition hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-slate-900 dark:hover:bg-slate-800 sm:min-w-[240px]"
        >
          <FileSpreadsheet className="h-5 w-5" />
          {downloading ? "Preparing CSV…" : "Download Cleaned CSV"}
        </button>
        <button
          type="button"
          onClick={handleDownload}
          className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-emerald-500 px-6 py-4 text-lg font-bold text-white shadow-lg transition hover:bg-emerald-600 sm:min-w-[240px]"
        >
          <Download className="h-5 w-5" />
          Download Serialized Model (.pkl)
        </button>
      </div>
    </section>
  );
}
