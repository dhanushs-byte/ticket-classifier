# Support Ticket Classification using DistilBERT and FastAPI

## Project Overview

This project classifies customer support tickets into categories and priority levels using Natural Language Processing (NLP) and Machine Learning. It compares a TF-IDF with Logistic Regression baseline against fine-tuned DistilBERT models and provides a REST API using FastAPI.

## Objectives

- Create a sample dataset containing 300 support tickets.
- Train a TF-IDF and Logistic Regression baseline model.
- Fine-tune DistilBERT for ticket category and priority classification.
- Compare models using accuracy, Macro F1, training time, and inference time.
- Develop and test REST API endpoints using FastAPI and Pydantic.
- Support individual and batch ticket predictions.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers and Datasets
- FastAPI and Uvicorn
- Pydantic
- Pytest
- Git and GitHub

## Project Structure

```text
ticket-classifier/
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── models/
│   ├── distilbert_category/
│   └── distilbert_priority/
├── results/
│   ├── baseline_results.csv
│   ├── distilbert_category_results.csv
│   ├── distilbert_priority_results.csv
│   └── model_comparison.csv
├── create_dataset.py
├── train_baseline.py
├── train_distilbert.py
├── train_priority.py
├── compare_models.py
├── app.py
├── test_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The project uses a synthetic dataset of 300 customer support tickets.

Each ticket contains:

- `ticket_id`: Unique ticket identifier.
- `text`: Customer support message.
- `category`: Billing, Technical Support, Account Access, Delivery, Product Inquiry, or Cancellation.
- `priority`: High, Medium, or Low.

The dataset is divided into training, validation, and test sets. The same splits are used for the baseline and DistilBERT evaluations.

**Note:** This synthetic dataset is intended for educational purposes. Its limited size and repeated text templates may produce results that do not generalize to real customer tickets.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/dhanushs-byte/ticket-classifier.git
cd ticket-classifier
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Project

### Step 1: Create the dataset

```bash
python create_dataset.py
```

This generates the training, validation, and test CSV files in the `data/` directory.

### Step 2: Train the baseline models

```bash
python train_baseline.py
```

The baseline uses TF-IDF text features and Logistic Regression to classify ticket categories and priorities.

### Step 3: Fine-tune DistilBERT for categories

```bash
python train_distilbert.py
```

### Step 4: Fine-tune DistilBERT for priority

```bash
python train_priority.py
```

The first DistilBERT training run requires an internet connection to download the pretrained model weights.

### Step 5: Generate the comparison table

```bash
python compare_models.py
```

The comparison results are saved in `results/model_comparison.csv`.

## Model Evaluation

The models are evaluated using:

- **Accuracy:** Overall proportion of correctly classified tickets.
- **Macro F1:** Average F1 score across classes, giving each class equal weight.
- **Training time:** Time required to train the model.
- **Inference time:** Average time required to predict one ticket.

### Recorded DistilBERT timings

| Metric | Category model | Priority model |
|---|---:|---:|
| Training time | 176.78 seconds | 237.85 seconds |
| Inference time | 33.43 ms/ticket | To be measured |

Actual accuracy and Macro F1 values should be taken from the saved evaluation results.

## Model Selection

TF-IDF with Logistic Regression provides a lightweight baseline with relatively low inference and resource costs. DistilBERT can capture contextual language patterns but requires more computation and memory.

The final deployment choice should be based on measured test accuracy, Macro F1, latency, and compute cost. For this small synthetic dataset, performance on real customer tickets must be evaluated before production use.

## FastAPI Application

The API loads the trained category and priority models at startup.

Start the server:

```bash
python -m uvicorn app:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### Health Check

**Endpoint:** `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy"
}
```

### Predict a Single Ticket

**Endpoint:** `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"I was charged twice\"}"
```

Example response structure:

```json
{
  "category": "Billing",
  "priority": "High",
  "confidence": 0.93
}
```

The example values are illustrative. Actual predictions depend on the trained models.

### Batch Prediction

**Endpoint:** `POST /predict-batch`

```bash
curl -X POST http://127.0.0.1:8000/predict-batch -H "Content-Type: application/json" -d "{\"tickets\":[{\"text\":\"I was charged twice\"},{\"text\":\"I cannot log in\"}]}"
```

The endpoint returns predictions for all supplied tickets.

### Input Validation

The API uses Pydantic to validate ticket requests. Empty or whitespace-only text returns HTTP `422 Unprocessable Entity`.

## Testing

Run the API tests:

```bash
python -m pytest -v test_app.py
```

The test suite covers:

- Health endpoint.
- Valid ticket prediction.
- Empty text validation.
- Whitespace-only text validation.
- Batch prediction.

## Deployment Decision: Five-Line Justification

1. DistilBERT can capture contextual meaning and may perform better on ambiguous support messages.
2. TF-IDF with Logistic Regression is lightweight, faster to train, and generally cheaper to serve.
3. Both models are evaluated using the same held-out test set to make the comparison fair.
4. The deployment decision should consider measured accuracy, Macro F1, inference latency, and infrastructure cost.
5. The selected model should meet the application's accuracy and speed requirements while remaining practical to maintain.

## Limitations and Future Improvements

- Expand the dataset with diverse, real-world labeled tickets.
- Evaluate performance on unseen customer messages.
- Tune hyperparameters and classification thresholds.
- Measure inference latency under concurrent API requests.
- Add logging, monitoring, and error handling.
- Containerize the application for deployment.

## Author

**Dhanush S**

GitHub: [dhanushs-byte](https://github.com/dhanushs-byte)
