"""
Train the Bangla health-text classifier for the Bangla Health Chatbot
(IEEE BECITHCON-2026).

Reads  : data/bangla_health_dataset.csv   (columns: text, category)
Writes : model/classifier.joblib  — fitted TF-IDF + LogisticRegression pipeline
         model/retrieval.joblib   — TF-IDF index of all dataset texts
         model/metrics.json       — held-out evaluation metrics (for the paper)

Run: python train_model.py
"""

import csv
import datetime
import json
import os
from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from nlp_engine import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "bangla_health_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_dataset():
    texts, labels = [], []
    with open(DATA_FILE, encoding="utf-8-sig") as f:  # utf-8-sig: file has a BOM
        for row in csv.DictReader(f):
            text = normalize(row["text"])
            label = row["category"].strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    return texts, labels


def build_pipeline():
    # Word n-grams capture Bangla vocabulary; char_wb n-grams make the model
    # robust to spelling variation and romanized Banglish.
    return Pipeline(
        [
            (
                "tfidf",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(2, 5),
                                sublinear_tf=True,
                                max_features=100_000,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def main():
    texts, labels = load_dataset()
    print(f"Loaded {len(texts)} samples from {os.path.relpath(DATA_FILE, BASE_DIR)}")
    print(f"Class distribution: {dict(Counter(labels).most_common())}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training on {len(X_train)} samples, evaluating on {len(X_test)}...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    class_order = sorted(set(labels))

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}\n")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(f"labels: {class_order}")
    for label, row in zip(class_order, confusion_matrix(y_test, y_pred, labels=class_order)):
        print(f"  {label:>10}: {row.tolist()}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    metrics = {
        "model": (
            "TF-IDF (word 1-2 grams + char_wb 2-5 grams) "
            "+ LogisticRegression(class_weight=balanced)"
        ),
        "dataset_file": os.path.relpath(DATA_FILE, BASE_DIR),
        "total_samples": len(texts),
        "class_distribution": dict(Counter(labels)),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix_labels": class_order,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=class_order).tolist(),
        "trained_at": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # The held-out split proved the approach; production model uses every sample.
    print("\nRefitting on the full dataset for production...")
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "classifier.joblib"), compress=3)

    # Retrieval index: vectors of every dataset entry, for similar-example lookup
    # at chat time. Queries are vectorized with the classifier's own tfidf step.
    vectorizer = pipeline.named_steps["tfidf"]
    joblib.dump(
        {
            "matrix": vectorizer.transform(texts),
            "texts": texts,
            "categories": labels,
        },
        os.path.join(MODEL_DIR, "retrieval.joblib"),
        compress=3,
    )

    print("Saved: model/classifier.joblib, model/retrieval.joblib, model/metrics.json")


if __name__ == "__main__":
    main()
