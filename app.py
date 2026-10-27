
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MAX_LENGTH = 128

predictors = {}


def load_classifier(target):
    path = MODEL_DIR / f"distilbert_{target}"

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    def predict(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        index = int(torch.argmax(probabilities).item())
        label = model.config.id2label.get(
            index,
            model.config.id2label.get(str(index), str(index)),
        )

        return str(label), float(probabilities[index].item())

    return predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictors["category"] = load_classifier("category")
    predictors["priority"] = load_classifier("priority")
    yield
    predictors.clear()


app = FastAPI(
    title="Support Ticket Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)


class TicketRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Ticket text cannot be empty")
        return value


class BatchRequest(BaseModel):
    tickets: list[TicketRequest]

    @field_validator("tickets")
    @classmethod
    def tickets_must_not_be_empty(cls, value):
        if not value:
            raise ValueError("Provide at least one ticket")
        return value


def classify(text):
    category, confidence = predictors["category"](text)
    priority, _ = predictors["priority"](text)

    return {
        "category": category,
        "priority": priority,
        "confidence": round(confidence, 4),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(ticket: TicketRequest):
    return classify(ticket.text)


@app.post("/predict-batch")
def predict_batch(request: BatchRequest):
    return {
        "predictions": [
            classify(ticket.text) for ticket in request.tickets
        ]
    }

