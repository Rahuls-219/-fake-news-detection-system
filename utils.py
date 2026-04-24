"""
utils.py
---------
Handles: text cleaning, model loading, prediction, confidence, explanation, and insight messages.
All functions are fast (<1s) and stateless — model is loaded once externally and passed in.
"""

import re
import string
import joblib
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
MODEL_PATH = Path("model/logistic_model.pkl")
VECTORIZER_PATH = Path("model/tfidf_vectorizer.pkl")

# ──────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────
# Simple stopwords (no NLTK download required at runtime)
STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "not", "with", "this", "that", "are", "was",
    "be", "by", "as", "from", "its", "they", "we", "he", "she", "you",
    "i", "my", "your", "our", "their", "has", "have", "had", "been",
    "will", "would", "could", "should", "do", "does", "did", "so",
    "if", "all", "can", "more", "about", "up", "out", "than", "then",
    "into", "after", "over", "also", "new", "just", "now", "years",
    "said", "says", "will", "one", "two", "three", "four", "five",
}


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers, strip stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
    text = re.sub(r"\d+", "", text)                      # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


# ──────────────────────────────────────────────
# MODEL LOADING (called once at startup)
# ──────────────────────────────────────────────
def load_model():
    """Load and return (model, vectorizer) tuple. Raises FileNotFoundError if not trained yet."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Please run `python train_model.py` first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
def predict(text: str, model, vectorizer) -> dict:
    """
    Given raw text, return prediction dict with:
      - label: 'FAKE' or 'REAL'
      - confidence: float 0–100
      - top_keywords: list of 3 strings
      - insight: human-readable explanation string
    """
    if not text or not text.strip():
        return None

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    # Prediction & probability
    label = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    fake_prob, real_prob = proba[0], proba[1]

    # Confidence = probability of the winning class
    confidence = max(fake_prob, real_prob) * 100

    # Top keywords from TF-IDF feature names
    top_keywords = get_top_keywords(vector, vectorizer, n=3)

    # Insight message
    insight = generate_insight(label, confidence, top_keywords, text)

    return {
        "label": label,
        "confidence": round(confidence, 1),
        "fake_prob": round(fake_prob * 100, 1),
        "real_prob": round(real_prob * 100, 1),
        "top_keywords": top_keywords,
        "insight": insight,
    }


# ──────────────────────────────────────────────
# KEYWORD EXTRACTION
# ──────────────────────────────────────────────
def get_top_keywords(vector, vectorizer, n: int = 3) -> list[str]:
    """Extract top-N TF-IDF weighted words from the input vector."""
    feature_names = vectorizer.get_feature_names_out()
    scores = vector.toarray()[0]
    top_indices = np.argsort(scores)[::-1][:n]
    keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
    return keywords if keywords else ["(no strong signals)"]


# ──────────────────────────────────────────────
# INSIGHT MESSAGE GENERATOR
# ──────────────────────────────────────────────
# Sensational trigger words commonly found in fake news
SENSATIONAL_WORDS = {
    "shocking", "exposed", "breaking", "alert", "urgent", "revealed",
    "secret", "banned", "deleted", "hidden", "suppressed", "leaked",
    "whistleblower", "conspiracy", "miracle", "cure", "cover", "truth",
    "won't believe", "share before", "wake up", "they don't want",
    "mainstream media", "big pharma", "new world order", "deep state",
}

CREDIBILITY_WORDS = {
    "study", "research", "report", "university", "scientists", "data",
    "official", "government", "published", "confirmed", "announced",
    "percent", "survey", "analysis", "findings", "review", "evidence",
}


def generate_insight(label: str, confidence: float, keywords: list, original_text: str) -> str:
    """Generate a simple, human-readable explanation."""
    text_lower = original_text.lower()

    sensational_hits = [w for w in SENSATIONAL_WORDS if w in text_lower]
    credibility_hits = [w for w in CREDIBILITY_WORDS if w in text_lower]

    kw_str = ", ".join(f'"{k}"' for k in keywords) if keywords else "unknown terms"

    if label == "FAKE":
        if confidence >= 85:
            strength = "strong signals"
        elif confidence >= 65:
            strength = "moderate signals"
        else:
            strength = "some signals"

        if sensational_hits:
            trigger = f"sensational language like '{sensational_hits[0]}'"
        else:
            trigger = f"unusual phrasing patterns"

        return (
            f"⚠️ This article shows {strength} of misinformation. "
            f"It contains {trigger} and key terms ({kw_str}) "
            f"that frequently appear in fabricated or misleading content. "
            f"Verify with trusted sources before sharing."
        )
    else:
        if credibility_hits:
            trigger = f"credible indicators like '{credibility_hits[0]}'"
        else:
            trigger = "neutral, factual language patterns"

        return (
            f"✅ This article appears to be legitimate news. "
            f"It uses {trigger} and key terms ({kw_str}) "
            f"consistent with verified reporting. "
            f"Always cross-check with multiple reliable sources."
        )
