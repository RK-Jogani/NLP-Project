"""Flask web application for single article Urdu news classification."""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

from flask import Flask, jsonify, render_template, request

from .paths import EVALUATION_PATH, MODEL_PATH
from .predict import load_model, predict_one

LABELS = {
    "کھیل": {"en": "Sports", "color": "#c65d3b", "icon": "◉"},
    "تفریح": {"en": "Entertainment", "color": "#9a68c7", "icon": "✳"},
    "کاروبار و معیشت": {"en": "Business & Economics", "color": "#347a69", "icon": "↗"},
    "سائنس اور ٹیکنالوجی": {"en": "Science & Technology", "color": "#477db3", "icon": "⌘"},
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


@lru_cache(maxsize=1)
def get_model():
    return load_model(MODEL_PATH)


def read_report():
    try:
        return json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@app.get("/")
def index():
    report = read_report()
    return render_template(
        "index.html",
        labels=LABELS,
        pipeline_ready=MODEL_PATH.is_file(),
        accuracy=report.get("accuracy"),
        train_rows=report.get("train_rows"),
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form
    article = payload.get("article", "")
    if not isinstance(article, str) or not article.strip():
        return jsonify(error="Please paste an Urdu news article first."), 400
    if len(article) > 12000:
        return jsonify(error="Please keep the article under 12,000 characters."), 400
    try:
        model = get_model()
        result = predict_one(model, article)
        label = result["prediction"]
        if label not in LABELS:
            app.logger.warning("Model returned an unexpected category: %s", label)
            return jsonify(error="The model returned an unsupported category."), 500

        raw_scores = result["scores"]
        values = list(raw_scores.values())
        if result["score_type"] == "probability":
            percentages = [value * 100 for value in values]
        elif values:
            low, high = min(values), max(values)
            percentages = [((value - low) / (high - low) * 100) if high > low else 100 / len(values)
                           for value in values]
        else:
            percentages = []
        scores = []
        for category, score in zip(raw_scores, percentages):
            info = LABELS.get(category)
            if info:
                scores.append({"ur": category, **info, "score": round(float(score), 1)})
        scores.sort(key=lambda item: item["score"], reverse=True)
        return jsonify(
            prediction={"ur": label, **LABELS[label]},
            scores=scores,
            score_type=result["score_type"],
            stats={
                "characters": len(article.strip()),
                "words": len(article.split()),
                "urdu_chars": sum("\u0600" <= char <= "\u06ff" for char in article),
            },
        )
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="The model could not classify this text. Please try again."), 500


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify(error="Request is too large."), 413


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG") == "1")
