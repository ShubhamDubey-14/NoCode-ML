import { Sparkles } from "lucide-react";

export default function DataPreview({
  analysis,
  targetColumn,
  setTargetColumn,
  imputationStrategy,
  setImputationStrategy,
  onTrain,
  training,
}) {
  if (!analysis) return null;

  const { columns, preview, row_count } = analysis;
  const colNames = columns.map((c) => c.name);

  return (
    <section className="mt-8 space-y-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold">Data preview</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {row_count.toLocaleString()} rows · {columns.length} columns
          </p>
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-slate-100 dark:bg-slate-800">
            <tr>
              {colNames.map((name) => (
                <th key={name} className="whitespace-nowrap px-3 py-2 font-semibold">
                  {name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, i) => (
              <tr key={i} className="border-t border-slate-200 dark:border-slate-700">
                {colNames.map((name) => (
                  <td key={name} className="whitespace-nowrap px-3 py-2 text-slate-600 dark:text-slate-300">
                    {row[name] ?? "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
            Columns
          </h3>
          <ul className="max-h-48 space-y-2 overflow-y-auto rounded-lg border border-slate-200 p-3 dark:border-slate-700">
            {columns.map((col) => (
              <li
                key={col.name}
                className="flex justify-between gap-2 text-sm"
              >
                <span className="font-medium">{col.name}</span>
                <span className="text-slate-500">
                  {col.dtype} · missing: {col.missing_count}
                </span>
              </li>
            ))}
          </ul>
        </div>

        <div className="space-y-4">
          <label className="block">
            <span className="mb-1 block text-sm font-semibold">Target column (y)</span>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 dark:border-slate-600 dark:bg-slate-800"
            >
              <option value="">Select target…</option>
              {colNames.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-sm font-semibold">Missing values</span>
            <select
              value={imputationStrategy}
              onChange={(e) => setImputationStrategy(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 dark:border-slate-600 dark:bg-slate-800"
            >
              <option value="mean_median">Impute (median / most frequent)</option>
              <option value="drop">Drop rows with missing values</option>
            </select>
          </label>
        </div>
      </div>

      <button
        type="button"
        disabled={!targetColumn || training}
        onClick={onTrain}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-accent px-6 py-4 text-lg font-bold text-white shadow-lg transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
      >
        <Sparkles className="h-5 w-5" />
        {training ? "Training AutoML engine…" : "Execute Automated AutoML Engine"}
      </button>
    </section>
  );
}
