import numpy as np
import pytest
from urdu_news_intelligence.preprocessing import normalize_text, extract_article, canonical_label, parse_split
from urdu_news_intelligence.predict import predict_one, read_batch_csv, load_model
import joblib

def test_urdu_normalization_preserves_content():
    assert normalize_text("  پاکستان،\n کھیل ") == "پاکستان، کھیل"
    assert normalize_text(None) == ""

def test_prompt_extraction_templates():
    assert extract_article("ہدایات پیراگراف: لاہور میں میچ ہوا۔") == "لاہور میں میچ ہوا۔"
    assert extract_article("ہدایات جملے: نئی خبر") == "نئی خبر"

@pytest.mark.parametrize("target,expected",[("یہ دی گی خبریں \"کھیل\" سے تعلق رکھتی ہیں","کھیل"),("یہ تفریح ہے","تفریح"),("Business & Economics","کاروبار و معیشت"),("x unknown","unknown")])
def test_category_mapping(target,expected):
    assert canonical_label(target)==(None if expected=="unknown" else expected)

def test_parse_reports_missing_unknown_and_duplicates(caplog):
    rows=[{"inputs":"پیراگراف: خبر","targets":"کھیل"},{"inputs":"پیراگراف: خبر","targets":"کھیل"},{"inputs":"","targets":"کھیل"},{"inputs":"خبر","targets":"نامعلوم"}]
    records,report=parse_split(rows)
    assert len(records)==1 and report["issues"]["duplicate_records"]==1
    assert report["issues"]["missing_or_empty_text"]==1
    assert report["issues"]["unexpected_or_missing_label"]==1

class Dummy:
    classes_=np.array(["کھیل","تفریح"])
    def predict(self,texts): return np.array(["کھیل"]*len(texts))
    def predict_proba(self,texts): return np.array([[.8,.2] for _ in texts])

def test_single_prediction_scores():
    result=predict_one(Dummy(),"خبر")
    assert result["prediction"]=="کھیل" and result["scores"]["کھیل"]==pytest.approx(.8)

def test_empty_prediction_rejected():
    with pytest.raises(ValueError): predict_one(Dummy(),"  ")

def test_batch_csv_validation():
    frame=read_batch_csv("article\nخبر\n\n".encode(),"news.csv")
    assert "article" in frame.columns and len(frame)==1
    with pytest.raises(ValueError): read_batch_csv(b"article\ntext", "news.xlsx")
    with pytest.raises(Exception): read_batch_csv(b"\xff\xfe", "bad.csv")

def test_model_loading(tmp_path):
    path=tmp_path/"model.joblib"
    joblib.dump({"pipeline":Dummy()},path)
    assert isinstance(load_model(path),Dummy)
    with pytest.raises(FileNotFoundError): load_model(tmp_path/"missing.joblib")
