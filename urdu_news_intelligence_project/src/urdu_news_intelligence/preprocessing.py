"""Urdu dataset parsing and text normalization utilities."""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)
LABELS = ["کھیل", "تفریح", "کاروبار و معیشت", "سائنس اور ٹیکنالوجی"]
LABEL_ALIASES = {
    "کھیل": "کھیل", "sports": "کھیل", "تفریح": "تفریح", "entertainment": "تفریح",
    "کاروبار و معیشت": "کاروبار و معیشت", "کاروبار اور معیشت": "کاروبار و معیشت",
    "business": "کاروبار و معیشت", "business & economics": "کاروبار و معیشت",
    "سائنس اور ٹیکنالوجی": "سائنس اور ٹیکنالوجی", "سائنس و ٹیکنالوجی": "سائنس اور ٹیکنالوجی",
    "science & technology": "سائنس اور ٹیکنالوجی", "science and technology": "سائنس اور ٹیکنالوجی",
}

def normalize_text(value: Any) -> str:
    """Normalize Unicode and whitespace while preserving Urdu letters and punctuation."""
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except Exception:
        pass
    text = unicodedata.normalize("NFC", str(value)).replace("\u0640", "")
    return re.sub(r"\s+", " ", text).strip()

def extract_article(prompt: Any) -> str:
    text = normalize_text(prompt)
    if not text:
        return ""
    # Actual dataset uses either paragraph: or sentences: (template_id 1 / 2).
    match = re.search(r"(?:پیراگراف|جملے|خبریں)\s*:\s*(.*)$", text)
    if match:
        text = match.group(1)
    return normalize_text(text).strip(" :،؛")

def canonical_label(target: Any) -> str | None:
    text = normalize_text(target).strip(" .،؛\"'“”")
    if not text:
        return None
    # The completion template wraps the label in Urdu quotation marks/words.
    for alias, label in sorted(LABEL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if text.casefold() == alias.casefold() or alias.casefold() in text.casefold():
            return label
    return None

def parse_split(split: Iterable[dict], split_name: str = "data", limit: int | None = None) -> tuple[list[dict], dict]:
    """Return valid records plus a quality report; unknown labels are counted and logged."""
    records, issues, seen = [], Counter(), set()
    for index, row in enumerate(split):
        if limit is not None and len(records) >= limit:
            break
        text = extract_article(row.get("inputs", row.get("text")))
        label = canonical_label(row.get("targets", row.get("label")))
        if not text:
            issues["missing_or_empty_text"] += 1
            continue
        if not label:
            issues["unexpected_or_missing_label"] += 1
            LOGGER.warning("%s row %d has an unrecognized label: %r", split_name, index, row.get("targets", row.get("label")))
            continue
        key = (text, label)
        if key in seen:
            issues["duplicate_records"] += 1
            continue
        seen.add(key)
        records.append({"text": text, "label": label, "source_index": index})
    report = {"source_rows_seen": index + 1 if 'index' in locals() else 0,
              "valid_records": len(records), "issues": dict(issues),
              "category_counts": dict(Counter(r["label"] for r in records))}
    return records, report
