
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

files = {
    "TF-IDF + Logistic Regression": (
        RESULT_DIR / "baseline_results.csv"
    ),
    "DistilBERT": (
        RESULT_DIR / "distilbert_category_results.csv"
    ),
}

records = []

for model_name, path in files.items():
    if not path.exists():
        print(f"Missing results file: {path}")
        continue

    df = pd.read_csv(path)

    # The comparison is for category classification.
    df = df[df["target"] == "category"]

    for _, row in df.iterrows():
        records.append({
            "Model": model_name,
            "Accuracy": row["accuracy"],
            "Macro F1": row["macro_f1"],
            "Training time (seconds)": row["training_time_seconds"],
            "Inference time (ms/ticket)": (
                row["inference_time_ms_per_ticket"]
                if "inference_time_ms_per_ticket" in df.columns
                else None
            ),
        })

if not records:
    raise RuntimeError(
        "No category results found. Run the baseline and "
        "DistilBERT training scripts first."
    )

comparison = pd.DataFrame(records)
output_path = RESULT_DIR / "model_comparison.csv"
comparison.to_csv(output_path, index=False)

print("\nCATEGORY MODEL COMPARISON")
print(comparison.to_string(index=False))
print("\nSaved to:", output_path)
