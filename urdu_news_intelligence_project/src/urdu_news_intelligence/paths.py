"""Canonical filesystem locations for source code and generated artifacts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODEL_PATH = MODEL_DIR / "urdu_news_classifier.joblib"
EVALUATION_PATH = REPORTS_DIR / "evaluation_report.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
MISCLASSIFIED_PATH = REPORTS_DIR / "misclassified.csv"
