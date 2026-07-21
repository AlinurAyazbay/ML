# SIS-3: Iris ML System with FastAPI + Streamlit + MLflow + Docker

Extension of **Practical Task 6**. This project turns the original FastAPI +
Docker deployment into a complete ML system with:

- **FastAPI** backend that serves predictions
- **Streamlit** frontend for user interaction
- **MLflow** tracking server for experiments + a Model Registry
- **Docker Compose** to orchestrate all three services

---

## Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Streamlit   │─────▶│   FastAPI    │      │   MLflow     │
│  (port 8501) │ HTTP │  (port 8000) │      │  (port 5000) │
└──────────────┘      └──────────────┘      └──────▲───────┘
                                                   │
                                            ┌──────┴───────┐
                                            │   train.py   │
                                            │ (logs runs)  │
                                            └──────────────┘
```

| Service   | Port  | Purpose                                            |
|-----------|-------|----------------------------------------------------|
| frontend  | 8501  | Streamlit UI — sliders, presets, probability chart |
| api       | 8000  | FastAPI — `/`, `/model`, `/predict`                |
| mlflow    | 5000  | MLflow tracking server + Model Registry UI         |

---

## Project Structure

```
sis3-ml-system/
│
├── api/
│   ├── main.py              # FastAPI app (2 required endpoints + /model)
│   ├── model.joblib         # Pretrained model (copy of the latest one)
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app.py               # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
│
├── training/
│   ├── train.py             # Training + MLflow logging + registry
│   └── requirements.txt
│
├── docker-compose.yml       # Orchestrates all three services
├── .gitignore
└── README.md                # This file
```

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
docker compose up --build
```

Then open:

- **Frontend (Streamlit):** http://localhost:8501
- **API docs (Swagger):** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000

To stop:

```bash
docker compose down
```

### Option B — Run each service locally

1. **MLflow server** (in terminal 1):
   ```bash
   pip install mlflow==2.17.0
   mlflow server --host 0.0.0.0 --port 5000 \
                 --backend-store-uri sqlite:///mlflow.db \
                 --default-artifact-root ./mlruns
   ```

2. **FastAPI** (in terminal 2):
   ```bash
   cd api
   pip install -r requirements.txt
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Streamlit** (in terminal 3):
   ```bash
   cd frontend
   pip install -r requirements.txt
   streamlit run app.py
   ```

---

## Training a New Model (MLflow experiment tracking)

The `training/train.py` script trains a Logistic Regression model and:

1. Creates/uses the MLflow experiment `iris-classification`
2. Logs hyperparameters (`C`, `max_iter`, `solver`, `test_size`, ...)
3. Logs evaluation metrics (`accuracy`, `f1_macro`, `precision_macro`, `recall_macro`)
4. Logs the model as an artifact
5. Registers the model as `iris-classifier` in the **Model Registry**
6. Writes a local `model.joblib` that the FastAPI service loads

### Run the default training

Make sure the MLflow server is running first, then:

```bash
cd training
pip install -r requirements.txt
python train.py
```

### Run with custom hyperparameters (creates a new registered version)

```bash
python train.py --C 0.5 --max-iter 300 --solver newton-cg
python train.py --C 2.0 --max-iter 500 --solver saga
```

Each run creates a **new version** of `iris-classifier` in the Model Registry,
which you can compare side-by-side in the MLflow UI.

### Deploy a new model to the FastAPI service

After training, copy the new `model.joblib` into the API folder:

```bash
cp training/model.joblib api/model.joblib
docker compose restart api
```

---

## API Endpoints

### `GET /`
Health check.
```json
{
  "message": "ML API is running",
  "model": "Logistic Regression",
  "dataset": "Iris",
  "status": "healthy"
}
```

### `GET /model`
Metadata about the loaded model.
```json
{
  "model_type": "LogisticRegression",
  "classes": ["setosa", "versicolor", "virginica"],
  "n_features": 4,
  "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
}
```

### `POST /predict`

Request:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response:
```json
{
  "species": "setosa",
  "species_id": 0,
  "confidence": 0.978,
  "probabilities": {
    "setosa": 0.978,
    "versicolor": 0.022,
    "virginica": 0.000
  }
}
```

### Test with curl

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

---

## MLflow — What the Grader Will See

After running training at least once, the MLflow UI at http://localhost:5000 shows:

- **Experiments tab:** `iris-classification` with one run per `python train.py` call.
  Each row shows the run's parameters and metrics. Click a run to see all
  hyperparameters, metrics, and artifacts (including the serialized model).
- **Models tab:** `iris-classifier` with multiple versions. Each version links
  back to the run that created it.

This satisfies both SIS-3 MLflow requirements:

| Requirement | Where it's done |
|---|---|
| Create an MLflow experiment | `mlflow.set_experiment("iris-classification")` |
| Log parameters | `mlflow.log_params(params)` |
| Log evaluation metrics | `mlflow.log_metrics({"accuracy": ..., "f1_macro": ...})` |
| Log model artifacts | `mlflow.sklearn.log_model(...)` + `mlflow.log_artifact("model.joblib")` |
| Register the model in the Model Registry | `registered_model_name="iris-classifier"` in `log_model(...)` |
| Assign a model name & version | Auto-assigned on each run (v1, v2, v3, ...) |

---

## Technologies Used

- **FastAPI** — modern web framework for the prediction API
- **Uvicorn** — ASGI server
- **Streamlit** — interactive Python web UI
- **MLflow** — experiment tracking + model registry
- **Scikit-learn** — Logistic Regression
- **Pydantic v2** — request/response validation
- **Docker + Docker Compose** — containerization & orchestration

---

## Author

Built on top of Practical Task 6 for the SIS-3 assignment.
