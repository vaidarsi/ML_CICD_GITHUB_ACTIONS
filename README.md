# CI/CD Pipeline for Machine Learning Using GitHub Actions

## 1. Project Overview

This project implements an end-to-end machine learning workflow using
Git, GitHub Actions, DVC, MLflow, Docker, FastAPI, Prometheus, and Grafana.

The application uses a Random Forest classifier to predict wine classes
from 13 numerical input features.

The pipeline automates data validation, preprocessing, model training,
automated testing, model quality validation, and Docker image building.
A separate GitHub Actions workflow handles container image publishing
and deployment verification.

## 2. Objectives

- Version-control source code using Git and GitHub.
- Manage the raw dataset using DVC.
- Train and evaluate a machine learning model.
- Track experiments and artifacts using MLflow.
- Validate model quality before deployment.
- Run automated tests using pytest.
- Build a Docker image for the prediction API.
- Automate CI/CD using GitHub Actions.
- Monitor API requests and performance using Prometheus and Grafana.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Machine learning and backend development |
| Pandas and NumPy | Data processing |
| Scikit-learn | Random Forest classification |
| DVC | Dataset versioning and retrieval |
| MLflow | Experiment tracking and model artifacts |
| FastAPI | Model prediction API |
| Pytest | Automated testing |
| Ruff | Python linting |
| Black | Code formatting checks |
| Docker | Containerization |
| GitHub Actions | Continuous integration and deployment |
| GitHub Container Registry | Docker image storage |
| Prometheus | Metrics collection |
| Grafana | Monitoring dashboards |

## 4. Project Structure

```text
ML_CICD_GITHUB_ACTIONS/
├── .dvc/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── notebooks/
├── reports/
├── scripts/
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── data_validation.py
│   ├── model_validation.py
│   ├── predict.py
│   └── train.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── params.yaml
├── requirements.txt
└── requirements-dev.txt
```

Generated files, reports, model artifacts, and additional files may be
created during training and execution.

## 5. Model Configuration

The project uses a Random Forest classifier.

Configuration is maintained in `params.yaml`.

Important parameters:

- Model type: Random Forest Classifier
- Number of estimators: 100
- Random state: 42
- Test size: 20%
- Target column: `target`
- Minimum required rows: 100

The model is evaluated using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score

## 6. Model Validation Gate

The project checks model metrics against minimum thresholds before
allowing the CI workflow to continue successfully.

Configured thresholds:

| Metric | Minimum threshold |
|---|---:|
| Accuracy | 0.90 |
| Precision | 0.90 |
| Recall | 0.90 |
| F1-score | 0.90 |

The validation script is:

`src/model_validation.py`

Run it using:

```bash
python -m src.model_validation
```

If a required metric is missing or falls below its threshold, the
validation step exits unsuccessfully. This prevents the CI workflow
from completing successfully.

## 7. Installation

### Prerequisites

Install the following software:

- Python 3.11
- Git
- Docker Desktop
- GitHub account
- Access to the configured DVC remote, when required

### Clone the repository

```bash
git clone https://github.com/vaidarsi/ML_CICD_GITHUB_ACTIONS.git
cd ML_CICD_GITHUB_ACTIONS
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## 8. Dataset Management Using DVC

The raw dataset is managed using DVC rather than being committed
directly to Git.

The repository contains a DVC-tracked dataset reference.

To retrieve the dataset from the configured DVC remote, use:

```bash
dvc pull -r dagshub
```

This command requires valid access to the configured remote.

In GitHub Actions, DVC remote credentials are supplied through GitHub
repository secrets:

- `DAGSHUB_USER`
- `DAGSHUB_TOKEN`

Never commit access tokens or passwords to the repository.

## 9. Data Validation and Preprocessing

Validate the dataset:

```bash
python -m src.data_validation
```

Run preprocessing:

```bash
python -m src.data_preprocessing
```

Preprocessing removes duplicate rows and rows containing missing values,
then writes the processed dataset to the configured output path.

## 10. Model Training and MLflow

Train the model:

```bash
python -m src.train
```

Training performs the following tasks:

1. Loads the processed dataset.
2. Separates the features and target.
3. Splits the dataset into training and test sets.
4. Trains the Random Forest classifier.
5. Calculates evaluation metrics.
6. Saves the trained model.
7. Generates evaluation reports.
8. Logs parameters, metrics, the model, and reports to MLflow.

The configured MLflow experiment name is:

`wine-classifier`

### Start the MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open:

http://127.0.0.1:5000

The MLflow UI displays experiments, runs, parameters, metrics, and
logged artifacts when the configured tracking directory contains them.

## 11. Automated Testing

Run the test suite:

```bash
python -m pytest -v
```

The tests cover API behavior, dataset validation, model training,
predictions, and model-validation logic.

The locally verified test run completed with:

`32 passed in 3.25s`

## 12. FastAPI Prediction Service

The API serves predictions from the trained model.

### Start the API locally

Ensure that the trained model exists, then run:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Application information |
| `/health` | GET | Model health status |
| `/predict` | POST | Generate a prediction |
| `/metrics` | GET | Expose Prometheus metrics |
| `/docs` | GET | Interactive Swagger documentation |

### Example prediction request

Send a POST request to:

`http://localhost:8000/predict`

JSON body:

```json
{
  "features": [
    13.2,
    1.78,
    2.14,
    11.2,
    100,
    2.65,
    2.76,
    0.26,
    1.28,
    4.38,
    1.05,
    3.4,
    1050
  ]
}
```

The request must contain exactly 13 numerical features in the expected
dataset column order.

Example response:

```json
{
  "prediction": "class_0",
  "model_version": "local-dev"
}
```

The prediction shown is an example response. Actual predictions depend
on the model and the supplied feature values.

## 13. Docker Deployment

Build the Docker image:

```bash
docker build -t wine-api:local .
```

Run the container:

```bash
docker run --rm -p 8000:8000 wine-api:local
```

Open the API documentation:

http://localhost:8000/docs

### Run the monitoring stack

Ensure Docker Desktop is running, then execute:

```bash
docker compose up -d --build
```

Check the container status:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f
```

Stop the Compose services when appropriate:

```bash
docker compose down
```

Only run `docker compose down` when you intend to stop this project's
services.

## 14. Continuous Integration

The CI workflow is defined in:

`.github/workflows/ci.yml`

It is configured to run for qualifying pushes to `main` and pull
requests targeting `main`.

The workflow includes:

1. Repository checkout.
2. Python setup.
3. Dependency installation.
4. Ruff lint checks.
5. Black formatting checks.
6. Dataset retrieval using DVC.
7. Data validation.
8. Data preprocessing.
9. Model training and MLflow logging.
10. Automated tests.
11. Model validation.
12. Artifact upload.
13. Docker image build check.

## 15. Continuous Deployment

The CD workflow is defined in:

`.github/workflows/cd.yml`

It runs after completion of the CI workflow and is configured to
continue only when the relevant CI run succeeds.

The deployment workflow:

1. Checks out the commit tested by CI.
2. Downloads the validated model artifact.
3. Verifies the model artifact exists.
4. Computes the Docker image name and tag.
5. Authenticates with GitHub Container Registry.
6. Builds and pushes the Docker image.
7. Starts a container in the GitHub Actions runner.
8. Checks the API health endpoint.
9. Tests the prediction endpoint.
10. Checks the metrics endpoint.

The GitHub Actions deployment verifies the container in its runner.
It does not, by itself, provide a permanently hosted public API.

## 16. Monitoring with Prometheus and Grafana

Prometheus collects metrics from the FastAPI `/metrics` endpoint.

The Prometheus configuration is located at:

`monitoring/prometheus.yml`

Grafana displays monitoring information through the configured dashboard.

### Local URLs

| Service | Address |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| MLflow | http://127.0.0.1:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

The Docker Compose configuration currently specifies Grafana's initial
admin username and password as `admin` and `admin`. Change the password
where appropriate and do not expose these development credentials
publicly.

### Useful Prometheus queries

Check whether the model is loaded:

```promql
model_loaded
```

Check HTTP request counters:

```promql
http_requests_total
```

Check total predictions:

```promql
sum(predictions_total)
```

Check HTTP errors:

```promql
http_errors_total
```

## 17. Verified Results

The following results were observed during local and GitHub Actions
verification:

- Local model training completed successfully.
- The model-validation gate passed with the observed metrics.
- MLflow displayed the `wine-classifier` experiment and artifacts.
- The local pytest run reported 32 passing tests.
- The Docker API returned a healthy status.
- The prediction endpoint returned a prediction.
- Prometheus reported the FastAPI target as UP.
- Grafana displayed API monitoring metrics.
- A GitHub Actions CI run completed successfully.
- A GitHub Actions CD run completed successfully.

These results describe the executions that were verified during
development; they do not guarantee future executions will always pass.

## 18. Future Improvements

Potential improvements include:

- Adding model signatures and input examples to MLflow model logging.
- Adding data-drift and model-drift monitoring.
- Improving automated retraining.
- Adding persistent storage for monitoring services.
- Deploying to a permanent cloud hosting environment.
- Adding stronger secrets management and deployment security.
- Adding integration tests for the deployed API.

## 19. Author

**Project:** CI/CD Pipeline for Machine Learning Using GitHub Actions

**Machine Learning Task:** Wine Classification