"""
train_model.py
---------------
Trains a Logistic Regression classifier on TF-IDF features.
Run: python train_model.py

Outputs:
  model/logistic_model.pkl
  model/tfidf_vectorizer.pkl
"""

import os
import sys
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Import our text cleaner
from utils import clean_text

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATA_PATH = Path("data/news_dataset.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "logistic_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

TFIDF_PARAMS = {
    "max_features": 10000,      # Vocabulary size cap
    "ngram_range": (1, 2),      # Unigrams + bigrams
    "min_df": 2,                # Ignore extremely rare words
    "max_df": 0.95,             # Ignore extremely common words
    "sublinear_tf": True,       # Apply log normalization to TF
}

LR_PARAMS = {
    "C": 1.0,                   # Regularization strength
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced", # Handle class imbalance
    "random_state": 42,
    "n_jobs": -1,               # Use all CPU cores
}


def load_data():
    """Load and validate dataset."""
    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found at {DATA_PATH}")
        print("  → Run: python generate_dataset.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    # Validate required columns
    if "text" not in df.columns or "label" not in df.columns:
        print("[ERROR] Dataset must have 'text' and 'label' columns.")
        sys.exit(1)

    # Drop nulls
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].str.upper().str.strip()

    # Keep only FAKE / REAL
    df = df[df["label"].isin(["FAKE", "REAL"])].reset_index(drop=True)

    print(f"[INFO] Dataset loaded: {len(df)} samples")
    print(f"       FAKE: {(df.label=='FAKE').sum()} | REAL: {(df.label=='REAL').sum()}")
    return df


def train():
    print("\n" + "="*50)
    print("  FAKE NEWS DETECTION — MODEL TRAINING")
    print("="*50)

    # 1. Load data
    df = load_data()

    # 2. Clean text
    print("\n[1/4] Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. TF-IDF Vectorization
    print("\n[2/4] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 5. Train Logistic Regression
    print("\n[3/4] Training Logistic Regression...")
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train_vec, y_train)

    # 6. Evaluate
    print("\n[4/4] Evaluating model...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n      Accuracy: {accuracy * 100:.2f}%")
    print("\n" + classification_report(y_test, y_pred))

    # 7. Save
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[✓] Model saved → {MODEL_PATH}")
    print(f"[✓] Vectorizer saved → {VECTORIZER_PATH}")
    print("\n[DONE] Training complete. Run: streamlit run app.py\n")

    return accuracy


if __name__ == "__main__":
    train()
