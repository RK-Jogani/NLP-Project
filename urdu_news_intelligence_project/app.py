import json
import os

import joblib
from flask import Flask, jsonify, render_template, request


ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "urdu_news_classifier.joblib")
REPORT_PATH = os.path.join(ROOT, "outputs", "evaluation_report.json")

LABELS = {
    "کھیل": {"en": "Sports", "color": "#c65d3b", "icon": "◉"},
    "تفریح": {"en": "Entertainment", "color": "#9a68c7", "icon": "✳"},
    "کاروبار و معیشت": {"en": "Business & Economics", "color": "#347a69", "icon": "↗"},
    "سائنس اور ٹیکنالوجی": {"en": "Science & Technology", "color": "#477db3", "icon": "⌘"},
}

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


def read_report():
    try:
        with open(REPORT_PATH, encoding="utf-8") as report_file:
            return json.load(report_file)
    except (OSError, ValueError):
        return {}


def get_pipeline():
    if not os.path.isfile(MODEL_PATH):
        return None
    model = joblib.load(MODEL_PATH)
    return model["pipeline"]


@app.get("/")
def index():
    pipeline_ready = os.path.isfile(MODEL_PATH)
    report = read_report()
    return render_template(
        "index.html",
        labels=LABELS,
        pipeline_ready=pipeline_ready,
        accuracy=report.get("accuracy"),
        train_rows=report.get("train_rows"),
        test_rows=report.get("test_rows"),
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form
    article = (payload.get("article") or "").strip()
    if not article:
        return jsonify(error="Please paste an Urdu news article first."), 400
    if len(article) > 12000:
        return jsonify(error="Please keep the article under 12,000 characters."), 400

    try:
        pipeline = get_pipeline()
        if pipeline is None:
            return jsonify(error="The trained model is missing. Run python train.py first."), 503
        prediction = str(pipeline.predict([article])[0])
        probabilities = pipeline.predict_proba([article])[0] if hasattr(pipeline, "predict_proba") else None
        scores = []
        if probabilities is not None:
            for label, score in zip(pipeline.classes_, probabilities):
                info = LABELS.get(str(label), {"en": str(label), "color": "#5c6b68", "icon": "•"})
                scores.append({"ur": str(label), "en": info["en"], "color": info["color"], "score": round(float(score) * 100, 1)})
            scores.sort(key=lambda item: item["score"], reverse=True)
        result = LABELS.get(prediction, {"en": prediction, "color": "#5c6b68", "icon": "•"})
        return jsonify(
            prediction={"ur": prediction, **result},
            scores=scores,
            stats={
                "characters": len(article),
                "words": len(article.split()),
                "urdu_chars": sum("\u0600" <= char <= "\u06ff" for char in article),
            },
        )
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="The model could not classify this text. Please try again."), 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host="127.0.0.1", port=5000)
