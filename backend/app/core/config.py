"""Application configuration and paths."""

import os
from pathlib import Path

# Base paths (Render: set MODEL_DIR via env for persistent disk if attached)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# CORS — Vercel frontend + local Vite dev
CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    os.getenv("FRONTEND_ORIGIN", "https://your-app.vercel.app"),
]

# Optuna trial cap (free-tier timeout guard)
OPTUNA_N_TRIALS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
