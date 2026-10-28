import argparse
import os
import re
import json
import joblib
import numpy as np
from datasets import load_dataset
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

DATASET_ID = "AhmadMustafa/Urdu-Instruct-News-Category-Classification"
LABELS = ["کھیل", "تفریح", "کاروبار و معیشت", "سائنس اور ٹیکنالوجی"]

def extract_article(prompt):
    """Remove the dataset's Urdu instruction template, retaining the article."""
    if not isinstance(prompt, str):
        return ""
    markers = ["پیراگراف:", "خبریں:", "جملے:", "پیراگراف"]
    for marker in markers:
        if marker in prompt:
            prompt = prompt.split(marker, 1)[-1]
            break
    # Remove common instruction text if it remains, then normalize whitespace.
    prompt = re.sub(r"اس پیراگراف.*?(?=پیراگراف:|خبریں:|جملے:)", " ", prompt)
    prompt = re.sub(r"دیے گی خبروں.*?(?=خبریں:|جملے:|پیراگراف:)", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip(" :۔،\t\n")
    return prompt

def extract_label(target):
    """Map target completion text to one of the four canonical Urdu labels."""
    if not isinstance(target, str):
        return None
    target = target.strip()
    # Prefer the longest labels first so substring matching is unambiguous.
    for label in sorted(LABELS, key=len, reverse=True):
        if label in target:
            return label
    # Common punctuation/spacing variants
    if "سائنس" in target or "ٹیکنالوجی" in target:
        return "سائنس اور ٹیکنالوجی"
    if "کاروبار" in target or "معیشت" in target:
        return "کاروبار و معیشت"
    if "تفریح" in target:
        return "تفریح"
    if "کھیل" in target:
        return "کھیل"
    return None

def prepare(split, limit=None):
    rows = []
    for item in split:
        text = extract_article(item.get("inputs", ""))
        label = extract_label(item.get("targets", ""))
        if text and label:
            rows.append((text, label))
        if limit and len(rows) >= limit:
            break
    return rows

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an Urdu news classifier.")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Optional cap for a quick trial, e.g. --max-train 12000")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Optional cap for a quick trial, e.g. --max-test 3000")
    args = parser.parse_args()

    print("Downloading dataset metadata/data from Hugging Face...")
    dataset = load_dataset(DATASET_ID)
    train_rows = prepare(dataset["train"], args.max_train)
    test_rows = prepare(dataset["test"], args.max_test)

    if not train_rows or not test_rows:
        raise RuntimeError("No usable rows were extracted. Inspect the dataset schema and prompt templates.")

    X_train = [x[0] for x in train_rows]
    y_train = [x[1] for x in train_rows]
    X_test = [x[0] for x in test_rows]
    y_test = [x[1] for x in test_rows]

    print(f"Usable training rows: {len(X_train):,}")
    print(f"Usable test rows: {len(X_test):,}")
    print("Training TF-IDF word + character features with Logistic Regression...")

    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                      min_df=2, max_features=180000, sublinear_tf=True)),
        ("char_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                      min_df=2, max_features=180000, sublinear_tf=True)),
    ])
    model = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, labels=LABELS,
                                   output_dict=True, zero_division=0)
    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=LABELS, zero_division=0))

    os.makedirs("models", exist_ok=True)
    joblib.dump({"pipeline": model, "labels": LABELS}, "models/urdu_news_classifier.joblib")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "classification_report": report,
                   "train_rows": len(X_train), "test_rows": len(X_test)},
                  f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_test, predictions, labels=LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Urdu News Classification — Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=180)
    plt.close(fig)

    print("\nSaved model: models/urdu_news_classifier.joblib")
    print("Saved metrics: outputs/evaluation_report.json")
    print("Saved confusion matrix: outputs/confusion_matrix.png")
    print("\nNext step: streamlit run app.py")

if __name__ == "__main__":
    main()
