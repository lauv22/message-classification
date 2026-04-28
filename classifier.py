"""
classifier.py
Message Classification System
─────────────────────────────
Trains a TF-IDF + Logistic Regression pipeline to classify messages
into Complaint, Feedback, or Inquiry.
"""

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dataset import DATA


# ── 1. Load Data ─────────────────────────────────────────────────────────────

def load_data():
    texts  = [item[0] for item in DATA]
    labels = [item[1] for item in DATA]
    return texts, labels


# ── 2. Build Pipeline ────────────────────────────────────────────────────────

def build_pipeline():
    """
    TF-IDF vectorizer converts raw text to numeric features.
    Logistic Regression performs multi-class classification.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=1,
            max_features=5000,
            sublinear_tf=True,    # apply log normalization to TF
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
        )),
    ])


# ── 3. Train & Evaluate ──────────────────────────────────────────────────────

def train_and_evaluate():
    texts, labels = load_data()

    # Split: 80% train, 20% test (stratified so each class is balanced)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print("=" * 56)
    print("  MESSAGE CLASSIFICATION SYSTEM — EVALUATION RESULTS")
    print("=" * 56)
    print(f"\n  Accuracy : {acc * 100:.1f}%")
    print(f"  Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Complaint", "Feedback", "Inquiry"]))

    # Cross-validation
    cv_scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
    print(f"  5-Fold CV Accuracy: {cv_scores.mean() * 100:.1f}% ± {cv_scores.std() * 100:.1f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["Complaint", "Feedback", "Inquiry"])
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print("              Complaint  Feedback  Inquiry")
    for label, row in zip(["Complaint", "Feedback ", "Inquiry  "], cm):
        print(f"  {label}   {row}")

    print("\n" + "=" * 56)

    # Save model
    model_path = Path(__file__).parent / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n  Model saved to: {model_path}")

    return pipeline


# ── 4. Inference ─────────────────────────────────────────────────────────────

def load_model():
    model_path = Path(__file__).parent / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run train_and_evaluate() first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def classify(message: str, pipeline=None) -> dict:
    """
    Classify a single message.

    Args:
        message: The text message to classify.
        pipeline: Optional pre-loaded pipeline. Loads from disk if not provided.

    Returns:
        dict with keys: category, confidence (per class)
    """
    if pipeline is None:
        pipeline = load_model()

    predicted_label = pipeline.predict([message])[0]
    probabilities   = pipeline.predict_proba([message])[0]
    classes         = pipeline.classes_

    confidence = {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, probabilities)}

    return {
        "message"   : message,
        "category"  : predicted_label,
        "confidence": confidence,
    }


# ── 5. Demo ──────────────────────────────────────────────────────────────────

def run_demo(pipeline):
    samples = [
        "Your platform has been down for hours and I'm losing sales!",
        "The new UI looks fantastic, love the cleaner layout.",
        "What is the maximum number of users allowed on the basic plan?",
        "I've been waiting three weeks for a refund and nobody is responding.",
        "Could you add dark mode to the mobile app?",
        "Do you support integration with Slack?",
    ]

    print("\n  SAMPLE PREDICTIONS")
    print("=" * 56)
    for msg in samples:
        result = classify(msg, pipeline)
        top_conf = result["confidence"][result["category"]]
        print(f"\n  Message   : {msg}")
        print(f"  Category  : {result['category']}  ({top_conf}% confidence)")
        conf_str = " | ".join(f"{k}: {v}%" for k, v in result["confidence"].items())
        print(f"  All scores: {conf_str}")
    print("\n" + "=" * 56)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = train_and_evaluate()
    run_demo(pipeline)
