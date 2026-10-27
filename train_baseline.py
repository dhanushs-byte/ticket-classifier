
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def load_data(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run create_dataset.py first."
        )

    df = pd.read_csv(path)

    required = {"text", "category", "priority"}
    if not required.issubset(df.columns):
        raise ValueError(f"{filename} must contain {required}")

    df = df.dropna(subset=["text", "category", "priority"]).copy()
    df["text"] = df["text"].astype(str)
    return df


train_df = load_data("train.csv")
val_df = load_data("validation.csv")
test_df = load_data("test.csv")

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))


def train_and_evaluate(target):
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    start = perf_counter()
    model.fit(train_df["text"], train_df[target])
    training_time = perf_counter() - start

    # Validation results
    val_predictions = model.predict(val_df["text"])
    val_accuracy = accuracy_score(
        val_df[target], val_predictions
    )
    val_f1 = f1_score(
        val_df[target],
        val_predictions,
        average="macro",
        zero_division=0,
    )

    # Final test results
    test_predictions = model.predict(test_df["text"])
    test_accuracy = accuracy_score(
        test_df[target], test_predictions
    )
    test_f1 = f1_score(
        test_df[target],
        test_predictions,
        average="macro",
        zero_division=0,
    )

    # Warm up before timing inference
    model.predict(["I was charged twice"])

    repetitions = 100
    start = perf_counter()

    for _ in range(repetitions):
        model.predict(["I was charged twice"])

    inference_ms = (
        (perf_counter() - start) * 1000 / repetitions
    )

    model_path = MODEL_DIR / f"{target}_model.joblib"
    joblib.dump(model, model_path)

    print(f"\n{'=' * 45}")
    print("Target:", target)
    print("Validation accuracy:", round(val_accuracy, 4))
    print("Validation macro F1:", round(val_f1, 4))
    print("Test accuracy:", round(test_accuracy, 4))
    print("Test macro F1:", round(test_f1, 4))
    print("Training time (seconds):", round(training_time, 4))
    print("Inference (ms/ticket):", round(inference_ms, 4))
    print("\nTest classification report:")
    print(classification_report(
        test_df[target],
        test_predictions,
        zero_division=0,
    ))
    print("Saved model:", model_path)

    return {
        "model": "TF-IDF + Logistic Regression",
        "target": target,
        "accuracy": test_accuracy,
        "macro_f1": test_f1,
        "training_time_seconds": training_time,
        "inference_time_ms_per_ticket": inference_ms,
    }


results = [
    train_and_evaluate("category"),
    train_and_evaluate("priority"),
]

results_df = pd.DataFrame(results)
results_df.to_csv(
    RESULT_DIR / "baseline_results.csv",
    index=False,
)

print("\nSaved metrics to results/baseline_results.csv")
print(results_df.to_string(index=False))
