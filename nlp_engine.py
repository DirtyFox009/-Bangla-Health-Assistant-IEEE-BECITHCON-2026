"""
NLP engine for the Bangla Health Chatbot - IEEE BECITHCON-2026.
Loads the TF-IDF + Logistic Regression classifier and the dataset
retrieval index produced by train_model.py.
"""

import os
import re

import joblib
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
CLASSIFIER_FILE = os.path.join(MODEL_DIR, "classifier.joblib")
RETRIEVAL_FILE = os.path.join(MODEL_DIR, "retrieval.joblib")

CATEGORY_LABELS_BN = {
    "symptoms": "লক্ষণ",
    "medicine": "ওষুধ",
    "nutrition": "পুষ্টি",
    "fever": "জ্বর",
    "emergency": "জরুরি",
    "doctor": "ডাক্তার পরামর্শ",
}

# Strong emergency markers force the category regardless of the classifier —
# only 100 emergency samples exist in training, so recall on them can't be
# trusted for a safety-critical label.
EMERGENCY_KEYWORDS = [
    "স্ট্রোক",
    "হার্ট অ্যাটাক",
    "হার্ট এটাক",
    "শ্বাসকষ্ট",
    "নিঃশ্বাস নিতে পারছ",
    "অজ্ঞান",
    "জ্ঞান হারিয়ে",
    "রক্তবমি",
    "খিঁচুনি",
    "বুকে তীব্র ব্যথা",
    "বিষ খেয়ে",
    "আত্মহত্যা",
    "রক্তক্ষরণ বন্ধ হচ্ছে না",
    "সাপে কেটেছে",
    "সাপে কামড়",
]

_classifier = None
_retrieval = None
_load_attempted = False


def normalize(text):
    """Normalization shared with training time (train_model.py imports this)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _load():
    global _classifier, _retrieval, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True
    try:
        _classifier = joblib.load(CLASSIFIER_FILE)
        _retrieval = joblib.load(RETRIEVAL_FILE)
        print(
            f"[nlp_engine] Loaded classifier + retrieval index "
            f"({len(_retrieval['texts'])} dataset entries)",
            flush=True,
        )
    except FileNotFoundError:
        print(
            "[nlp_engine] WARNING: model artifacts not found — run `python train_model.py` "
            "first. Chatbot will answer without classification/retrieval.",
            flush=True,
        )


def _is_emergency(text):
    t = normalize(text)
    return any(kw in t for kw in EMERGENCY_KEYWORDS)


def predict_category(text):
    """Return (category, confidence). (None, 0.0) if the model is unavailable."""
    if _is_emergency(text):
        return "emergency", 1.0
    _load()
    if _classifier is None:
        return None, 0.0
    proba = _classifier.predict_proba([normalize(text)])[0]
    idx = proba.argmax()
    return _classifier.classes_[idx], float(proba[idx])


def retrieve_similar(text, k=3):
    """Top-k most similar dataset entries: [{text, category, similarity}]."""
    _load()
    if _classifier is None or _retrieval is None:
        return []
    query = _classifier.named_steps["tfidf"].transform([normalize(text)])
    sims = cosine_similarity(query, _retrieval["matrix"])[0]
    top = sims.argsort()[::-1][:k]
    return [
        {
            "text": _retrieval["texts"][i],
            "category": _retrieval["categories"][i],
            "similarity": float(sims[i]),
        }
        for i in top
    ]
