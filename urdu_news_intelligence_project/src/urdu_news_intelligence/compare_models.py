"""Compare classical baselines on training-only CV, then evaluate once on official test."""
import argparse, json, time
from pathlib import Path
import joblib
from datasets import load_dataset
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from .paths import MODEL_PATH, REPORTS_DIR, COMPARISON_PATH
from .preprocessing import LABELS, parse_split
from .evaluate import evaluate_predictions, save_evaluation
def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=120000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=120000, sublinear_tf=True)),
    ])

def candidates():
    return {"Multinomial Naive Bayes": MultinomialNB(alpha=.5),
            "Logistic Regression": LogisticRegression(max_iter=1000, C=4, class_weight="balanced"),
            "Linear SVM": LinearSVC(C=1, class_weight="balanced")}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--max-train",type=int); p.add_argument("--max-test",type=int); p.add_argument("--cv-folds",type=int,default=3); args=p.parse_args()
    ds=load_dataset("AhmadMustafa/Urdu-Instruct-News-Category-Classification")
    train, train_quality=parse_split(ds["train"],"train",args.max_train); test,test_quality=parse_split(ds["test"],"test",args.max_test)
    X=[r["text"] for r in train]; y=[r["label"] for r in train]; Xt=[r["text"] for r in test]; yt=[r["label"] for r in test]
    if len(set(y))<4 or not test: raise RuntimeError("Need all four classes in training and a non-empty official test set.")
    cv=StratifiedKFold(n_splits=args.cv_folds,shuffle=True,random_state=42); rows=[]
    for name, clf in candidates().items():
        pipe=Pipeline([("features",features()),("classifier",clf)])
        scores=cross_validate(pipe,X,y,cv=cv,scoring={"macro_f1":"f1_macro","accuracy":"accuracy","weighted_f1":"f1_weighted"},n_jobs=1)
        rows.append({"model":name,"cv_macro_f1":scores["test_macro_f1"].mean(),"cv_accuracy":scores["test_accuracy"].mean(),"cv_weighted_f1":scores["test_weighted_f1"].mean()})
    winner=max(rows,key=lambda r:r["cv_macro_f1"])["model"]
    chosen=Pipeline([("features",features()),("classifier",candidates()[winner])]); start=time.perf_counter(); chosen.fit(X,y); elapsed=time.perf_counter()-start
    pred=chosen.predict(Xt); model_path=MODEL_PATH; model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"pipeline":chosen,"labels":LABELS,"model_name":winner},model_path)
    result=evaluate_predictions(yt,pred,winner,len(X),len(Xt),elapsed,model_path.stat().st_size)
    train_texts={r["text"] for r in train}; test_texts={r["text"] for r in test}
    overlap=train_texts & test_texts
    result.update({"train_quality":train_quality,"test_quality":test_quality,
                   "cross_split_exact_duplicate_count":len(overlap),
                   "cross_split_exact_duplicate_examples":[t[:300] for t in list(overlap)[:10]],
                   "selection_metric":"training-only stratified CV macro-F1","cv_results":rows})
    save_evaluation(result,yt,pred,Xt)
    COMPARISON_PATH.parent.mkdir(parents=True,exist_ok=True)
    COMPARISON_PATH.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"selected_model":winner,"cv_results":rows,"official_test":{k:result[k] for k in ("accuracy","macro_f1","weighted_f1","test_rows")}},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
