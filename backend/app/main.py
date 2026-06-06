"""FastAPI entrypoint — CSV analysis, AutoML training, model download."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from app.core.automl import train_automl_pipeline
from app.core.config import CORS_ORIGINS, MODEL_DIR
from app.core.preprocessing import clean_dataframe

app = FastAPI(
    title="No-Code Data Cleaning & AutoML API",
    description="Upload CSV, clean data, train classifiers, download .pkl pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_csv(upload: UploadFile) -> pd.DataFrame:
    if not upload.filename or not upload.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """Return column metadata, missing counts, and a 5-row preview."""
    df = _read_csv(file)
    columns = []
    for col in df.columns:
        columns.append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
            }
        )
    preview = df.head(5).fillna("").astype(str).to_dict(orient="records")
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "preview": preview,
    }


@app.post("/api/process-and-train")
async def process_and_train(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    imputation_strategy: str = Form(...),
    scaling_method: str = Form(default="none"),
) -> dict[str, Any]:
    """Preprocess data, run AutoML, persist pipeline, return metrics."""
    if imputation_strategy not in ("mean_median", "drop"):
        raise HTTPException(
            status_code=400,
            detail="imputation_strategy must be 'mean_median' or 'drop'.",
        )
    if scaling_method not in ("none", "standardize", "normalize"):
        raise HTTPException(
            status_code=400,
            detail="scaling_method must be 'none', 'standardize', or 'normalize'.",
        )
    df = _read_csv(file)
    try:
        result = train_automl_pipeline(df, target_column, imputation_strategy, scaling_method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc
    return result


@app.post("/api/download-cleaned-csv")
async def download_cleaned_csv(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    imputation_strategy: str = Form(...),
    scaling_method: str = Form(default="none"),
) -> Response:
    """Return cleaned/imputed dataset as CSV (same rules as training preprocessing).
    
    Note: Downloaded CSV is unscaled for human readability. Scaling is only applied during model training.
    """
    if imputation_strategy not in ("mean_median", "drop"):
        raise HTTPException(
            status_code=400,
            detail="imputation_strategy must be 'mean_median' or 'drop'.",
        )
    if scaling_method not in ("none", "standardize", "normalize"):
        raise HTTPException(
            status_code=400,
            detail="scaling_method must be 'none', 'standardize', or 'normalize'.",
        )
    df = _read_csv(file)
    try:
        cleaned = clean_dataframe(df, target_column, imputation_strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
    base = (file.filename or "dataset").rsplit(".", 1)[0]
    filename = f"{base}_cleaned.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/download-model/{model_id}")
def download_model(model_id: str) -> FileResponse:
    """Stream the saved joblib pipeline (.pkl) to the client."""
    safe_id = model_id.strip()
    if not safe_id or ".." in safe_id or "/" in safe_id or "\\" in safe_id:
        raise HTTPException(status_code=400, detail="Invalid model_id.")

    path = MODEL_DIR / f"{safe_id}.pkl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Model not found.")

    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"model_{safe_id}.pkl",
    )
