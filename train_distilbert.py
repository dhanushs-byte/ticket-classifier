
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "distilbert_category"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 128
EPOCHS = 2
BATCH_SIZE = 8


def load_data(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run create_dataset.py first."
        )

    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    return df.reset_index(drop=True)


train_df = load_data("train.csv")
val_df = load_data("validation.csv")
test_df = load_data("test.csv")

labels = sorted(train_df["category"].unique().tolist())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

print("Training examples:", len(train_df))
print("Validation examples:", len(val_df))
print("Test examples:", len(test_df))
print("Categories:", labels)
print("Loading pretrained DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def prepare_dataset(df):
    encoded = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    return Dataset.from_dict({
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": [
            label2id[label] for label in df["category"].tolist()
        ],
    })


train_dataset = prepare_dataset(train_df)
val_dataset = prepare_dataset(val_df)
test_dataset = prepare_dataset(test_df)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)


def compute_metrics(eval_prediction):
    logits, true_labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


training_args = TrainingArguments(
    output_dir=str(BASE_DIR / "results" / "training_checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    report_to="none",
    seed=42,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting fine-tuning...")
training_start = perf_counter()
trainer.train()
training_seconds = perf_counter() - training_start

print("\nValidation results:")
print(trainer.evaluate(val_dataset))

print("\nTest results:")
test_metrics = trainer.evaluate(test_dataset)
print(test_metrics)

test_output = trainer.predict(test_dataset)
test_predictions = np.argmax(test_output.predictions, axis=-1)
test_labels = test_output.label_ids

print("\nTest classification report:")
print(classification_report(
    test_labels,
    test_predictions,
    labels=list(range(len(labels))),
    target_names=labels,
    zero_division=0,
))

# Save the trained model and tokenizer.
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

# Warm up the model before timing inference.
device = next(model.parameters()).device
sample = tokenizer(
    "I was charged twice",
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH,
).to(device)

model.eval()
with torch.no_grad():
    for _ in range(3):
        model(**sample)

if device.type == "cuda":
    torch.cuda.synchronize()

repetitions = 30
start = perf_counter()

with torch.no_grad():
    for _ in range(repetitions):
        model(**sample)

if device.type == "cuda":
    torch.cuda.synchronize()

inference_ms = (perf_counter() - start) * 1000 / repetitions

result = {
    "model": "DistilBERT",
    "target": "category",
    "accuracy": test_metrics["eval_accuracy"],
    "macro_f1": test_metrics["eval_macro_f1"],
    "training_time_seconds": training_seconds,
    "inference_time_ms_per_ticket": inference_ms,
}

result_path = RESULT_DIR / "distilbert_category_results.csv"
pd.DataFrame([result]).to_csv(result_path, index=False)

print("\nTraining completed!")
print("Training time (seconds):", round(training_seconds, 2))
print("Inference time (ms/ticket):", round(inference_ms, 2))
print("Model saved at:", MODEL_DIR)
print("Metrics saved at:", result_path)
