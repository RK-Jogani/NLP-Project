# Urdu News Desk — University NLP Project

A complete, locally runnable Urdu news category classification project built with a classical NLP baseline and a responsive editorial-style Flask interface.

## Dataset

Hugging Face: [AhmadMustafa/Urdu-Instruct-News-Category-Classification](https://huggingface.co/datasets/AhmadMustafa/Urdu-Instruct-News-Category-Classification)

The dataset card lists 100,674 training rows and 11,187 test rows. Each row contains an Urdu instruction prompt (`inputs`) and a completion (`targets`). This project extracts the news text from the prompt and maps target completions to four canonical labels:

- کھیل — Sports
- تفریح — Entertainment
- کاروبار و معیشت — Business & Economics
- سائنس اور ٹیکنالوجی — Science & Technology

## Features

- Urdu text input and category prediction
- Responsive Flask newsroom interface with live category scores
- Word-level TF-IDF bigrams and character-level TF-IDF n-grams
- Logistic Regression classifier with class balancing
- Evaluation on the dataset's provided test split
- Accuracy, precision, recall, F1-score, confusion matrix
- Saved model for local inference

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python app.py
```

First training downloads the dataset from Hugging Face and can take several minutes depending on internet speed, RAM and CPU. The full dataset is recommended for final evaluation.

### Quick smoke test

To test the pipeline on a smaller subset before running the full training:

```powershell
python train.py --max-train 12000 --max-test 3000
python app.py
```

The quick run is for testing the workflow only; do not report its metrics as full-dataset results.

## Methodology

1. Load the official train and test splits.
2. Extract article text from the Urdu prompt templates.
3. Normalize the four target classes.
4. Represent text with word TF-IDF (1–2 grams) and character TF-IDF (2–5 grams).
5. Train Logistic Regression.
6. Evaluate predictions against the held-out test split.
7. Save the trained pipeline and evaluation outputs.

The official test split is used only for final evaluation, not for fitting the model. For a stronger study, additionally audit duplicate/near-duplicate articles and report per-class results. If duplicates cross splits, performance may be optimistic.

## Project structure

```text
urdu_news_intelligence/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── models/
└── outputs/
```

`models/` and `outputs/` are populated after training.

## Academic extensions

For a stronger university submission, compare this baseline against Multinomial Naive Bayes and a fine-tuned multilingual transformer such as XLM-R. Report macro-F1, weighted-F1, per-class precision/recall, training time and model size. Keep the same held-out test split and document all hyperparameters.

## Limitations

- The classifier predicts a topic category; it does not verify whether a news story is true.
- Model scores are not guaranteed to be calibrated probabilities.
- Urdu spelling variation, code-switching, very short headlines and topics outside the four classes may reduce performance.
- Dataset labels and source distribution may reflect the original dataset's coverage and annotation decisions.
- Review the dataset's current license and attribution requirements on Hugging Face before redistributing data or a trained model.

## Dataset citation

AhmadMustafa. *Urdu-Instruct-News-Category-Classification*. Hugging Face Datasets. https://huggingface.co/datasets/AhmadMustafa/Urdu-Instruct-News-Category-Classification

The dataset card notes that this dataset was converted from work by Khalid Hussain, Nimra Mughal, Irfan Ali, Saif Hassan, and Sher Muhammad Daudpota. Please consult the dataset card for original attribution and license information.
