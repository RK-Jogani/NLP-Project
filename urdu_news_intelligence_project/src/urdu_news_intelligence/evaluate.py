"""Evaluation artifacts for a trained model on the official test split."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .paths import REPORTS_DIR
from .preprocessing import LABELS

def evaluate_predictions(y_true, y_pred, model_name="model", train_rows=None, test_rows=None, training_seconds=None, model_bytes=None):
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {"model": model_name, "accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": report["macro avg"]["precision"], "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"], "weighted_f1": report["weighted avg"]["f1-score"],
            "classification_report": report, "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
            "labels": LABELS, "train_rows": train_rows, "test_rows": test_rows,
            "evaluated_examples": len(y_true), "training_seconds": training_seconds, "model_bytes": model_bytes}

def save_evaluation(result, y_true, y_pred, texts, output_dir=REPORTS_DIR):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    errors = pd.DataFrame({"text": texts, "true_label": y_true, "predicted_label": y_pred}).query("true_label != predicted_label")
    errors.to_csv(output_dir / "misclassified.csv", index=False, encoding="utf-8-sig")
    result["common_errors"] = [{"true_label": a, "predicted_label": b, "count": n}
                               for (a, b), n in Counter(zip(errors.true_label, errors.predicted_label)).most_common(10)]
    result["sample_records"] = [{"text": t[:450], "label": y} for t, y in list(zip(texts, y_true))[:12]]
    (output_dir / "evaluation_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    cm = result["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8, 6)); im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right"); ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title("Official test split confusion matrix")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)): ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=180); plt.close(fig)

if __name__ == "__main__":
    raise SystemExit("Evaluation is produced by train.py/compare_models.py so test data remains held out.")
