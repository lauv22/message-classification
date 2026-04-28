"""
predict.py
Interactive command-line interface to test the trained classifier.
Usage:
    python predict.py                  # interactive mode
    python predict.py "your message"   # single prediction
"""

import sys
import pickle
from pathlib import Path


def load_model():
    model_path = Path(__file__).parent / "model.pkl"
    if not model_path.exists():
        print("Model not found. Please run: python classifier.py")
        sys.exit(1)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(message, pipeline):
    label = pipeline.predict([message])[0]
    probs = pipeline.predict_proba([message])[0]
    classes = pipeline.classes_
    confidence = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)}
    return label, confidence


def print_result(message, label, confidence):
    bar = {"Complaint": "🔴", "Feedback": "🔵", "Inquiry": "🟢"}
    print(f"\n  Input    : {message}")
    print(f"  Category : {bar.get(label, '•')} {label}  ({confidence[label]}% confidence)")
    print("  Scores   :", " | ".join(f"{k}: {v}%" for k, v in confidence.items()))


def main():
    pipeline = load_model()

    # Single prediction from CLI argument
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        label, conf = predict(message, pipeline)
        print_result(message, label, conf)
        return

    # Interactive mode
    print("\n  Message Classifier — Interactive Mode")
    print("  Type a message and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            message = input("  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break
        if not message:
            continue
        if message.lower() in ("quit", "exit", "q"):
            break
        label, conf = predict(message, pipeline)
        print_result(message, label, conf)
        print()


if __name__ == "__main__":
    main()
