import { useCallback, useRef, useState } from "react";
import { CheckCircle2, FileSpreadsheet, Upload, XCircle } from "lucide-react";

export default function Dropzone({ onFileAccepted, disabled }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState("idle"); // idle | uploading | done | error

  const validate = (f) => {
    if (!f) return false;
    if (!f.name.toLowerCase().endsWith(".csv")) {
      setError("Only .csv files are supported.");
      setProgress("error");
      return false;
    }
    setError("");
    return true;
  };

  const handleFile = useCallback(
    async (f) => {
      if (!validate(f)) return;
      setFile(f);
      setProgress("uploading");
      try {
        await onFileAccepted(f);
        setProgress("done");
      } catch (e) {
        setError(e.message || "Upload failed");
        setProgress("error");
      }
    },
    [onFileAccepted]
  );

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    const f = e.dataTransfer.files?.[0];
    handleFile(f);
  };

  return (
    <div className="space-y-3">
      <div
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 transition ${
          dragOver
            ? "border-accent bg-indigo-50 dark:bg-indigo-950/30"
            : "border-slate-300 bg-white dark:border-slate-600 dark:bg-slate-900"
        } ${disabled ? "pointer-events-none opacity-60" : ""}`}
      >
        <Upload className="mb-3 h-10 w-10 text-accent" />
        <p className="text-lg font-semibold">Drop your CSV here</p>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          or click to browse — .csv only
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      {file && (
        <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-700 dark:bg-slate-900">
          <FileSpreadsheet className="h-5 w-5 text-emerald-600" />
          <div className="flex-1 min-w-0">
            <p className="truncate font-medium">{file.name}</p>
            <p className="text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB</p>
          </div>
          {progress === "done" && (
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-1 text-xs font-medium text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
              <CheckCircle2 className="h-3.5 w-3.5" /> Validated
            </span>
          )}
          {progress === "uploading" && (
            <span className="text-xs font-medium text-accent animate-pulse">Analyzing…</span>
          )}
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
          <XCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}
    </div>
  );
}
