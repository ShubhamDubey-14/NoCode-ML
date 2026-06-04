import { useState } from "react";
import { BarChart3 } from "lucide-react";
import { analyzeCsv, downloadCleanedCsv, processAndTrain } from "./api.js";
import ThemeToggle from "./components/ThemeToggle.jsx";
import Dropzone from "./components/Dropzone.jsx";
import DataPreview from "./components/DataPreview.jsx";
import DashboardMetrics from "./components/DashboardMetrics.jsx";

export default function App() {
  const [file, setFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [imputationStrategy, setImputationStrategy] = useState("mean_median");
  const [trainResult, setTrainResult] = useState(null);
  const [training, setTraining] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [globalError, setGlobalError] = useState("");

  const handleFileAccepted = async (f) => {
    setFile(f);
    setTrainResult(null);
    setGlobalError("");
    const data = await analyzeCsv(f);
    setAnalysis(data);
    if (data.columns?.length) {
      setTargetColumn(data.columns[data.columns.length - 1].name);
    }
  };

  const handleDownloadCleaned = async () => {
    if (!file || !targetColumn) return;
    setDownloading(true);
    setGlobalError("");
    try {
      await downloadCleanedCsv(file, targetColumn, imputationStrategy);
    } catch (e) {
      setGlobalError(e.message || "CSV download failed");
    } finally {
      setDownloading(false);
    }
  };

  const handleTrain = async () => {
    if (!file || !targetColumn) return;
    setTraining(true);
    setGlobalError("");
    try {
      const result = await processAndTrain(file, targetColumn, imputationStrategy);
      setTrainResult(result);
    } catch (e) {
      setGlobalError(e.message || "Training failed");
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 transition-colors duration-300 dark:bg-slate-950">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-900/80">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <div className="flex items-center gap-3">
            <BarChart3 className="h-8 w-8 text-accent" />
            <div>
              <h1 className="text-xl font-bold tracking-tight">No-Code AutoML</h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Data cleaning · Automated ML · Model export
              </p>
            </div>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-10">
        <Dropzone onFileAccepted={handleFileAccepted} disabled={training} />

        {globalError && (
          <p className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
            {globalError}
          </p>
        )}

        <DataPreview
          analysis={analysis}
          targetColumn={targetColumn}
          setTargetColumn={setTargetColumn}
          imputationStrategy={imputationStrategy}
          setImputationStrategy={setImputationStrategy}
          onTrain={handleTrain}
          training={training}
        />

        <DashboardMetrics
          result={trainResult}
          onDownloadCleaned={handleDownloadCleaned}
          downloading={downloading}
        />
      </main>
    </div>
  );
}
