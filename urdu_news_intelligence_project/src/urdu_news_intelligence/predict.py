"""Local inference helpers for the saved scikit-learn model."""
from pathlib import Path
import io
import joblib
import pandas as pd
from .paths import MODEL_PATH
from .preprocessing import normalize_text


def read_batch_csv(data: bytes, filename: str) -> pd.DataFrame:
    """Read a bounded-format CSV/TSV payload with common Urdu encodings."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".tsv"}:
        raise ValueError("Upload a .csv or .tsv file.")
    sep = "\t" if suffix == ".tsv" else ","
    error = None
    encodings = ("utf-8-sig", "utf-16", "cp1256", "utf-8")
    for encoding in encodings:
        try:
            frame = pd.read_csv(io.BytesIO(data), sep=sep, encoding=encoding, on_bad_lines="error")
            if len(frame.columns) == 0:
                raise ValueError("The file has no header columns.")
            return frame
        except UnicodeDecodeError as exc:
            error = exc
    raise ValueError(f"Could not decode the upload as UTF-8 or Windows Arabic encoding: {error}")

def load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}. Run python train.py first.")
    artifact = joblib.load(path)
    return artifact.get("pipeline", artifact) if isinstance(artifact, dict) else artifact

def predict_articles(model, articles):
    cleaned = [normalize_text(article) for article in articles]
    if any(not text for text in cleaned):
        raise ValueError("Every article must contain non-empty text.")
    predictions = model.predict(cleaned)
    if hasattr(model, "predict_proba"):
        scores, score_type = model.predict_proba(cleaned), "probability"
    elif hasattr(model, "decision_function"):
        scores, score_type = model.decision_function(cleaned), "decision score"
    else:
        scores, score_type = None, None
    result = []
    for i, label in enumerate(predictions):
        row = {"prediction": str(label), "score_type": score_type, "scores": {}}
        if scores is not None:
            values = scores[i]
            if getattr(values, "ndim", 0) == 0:
                values = [float(values)]
            row["scores"] = {str(c): float(v) for c, v in zip(model.classes_, values)}
        result.append(row)
    return result

def predict_one(model, article):
    return predict_articles(model, [article])[0]
