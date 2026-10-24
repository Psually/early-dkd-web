# Early Kidney Disease Prediction in Adults with Obesity and Type 2 Diabetes

## Introduction

This repository contains a research web application for identifying patients with the study-defined early kidney disease outcome. A frozen random-forest model uses **31 clinical predictors**: 17 selected clinical and blood indicators, six urine indicators, and eight TyG-related metabolic indicators. The application returns an estimated probability and an individual SHAP waterfall explanation.

The tool supports research and follow-up assessment. Its output is not a clinical diagnosis.

## Repository structure

| Directory | Contents |
| --- | --- |
| [`app/`](app/) | Prediction service and English web interface |
| [`models/`](models/) | Frozen model, version information, and model checksum |
| [`scripts/`](scripts/) | Model reconstruction, verification, and synthetic self-test |
| [`docs/`](docs/) | Predictor definitions, methods, audits, and upload notes |

## Run the web application

Use Python 3.12 from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.self_test
.\.venv\Scripts\python.exe -m app.serve
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/). On Windows, `start_web.cmd` or `run.ps1` starts the same local service. The service binds to `127.0.0.1` and does not save submitted measurements or predictions.

## Public deployment

The public research application is available at [https://psually.pythonanywhere.com/](https://psually.pythonanywhere.com/). It is hosted on PythonAnywhere's free Beginner plan. The account owner must renew the free web app in the PythonAnywhere Web tab once per month.

`render.yaml` defines a Python web service for Render. Create a Render Blueprint from this GitHub repository to deploy it. The deployment runs the synthetic self-test during the build, listens on Render's `PORT` at `0.0.0.0`, and uses `/api/health` for health checks. Render provides the public HTTPS URL after deployment. Enter only de-identified measurements in the public research tool; the app does not persist request values or predictions.

PythonAnywhere can run the same application through `app.wsgi:application`. From a Python 3.12 virtual environment, install `requirements.txt` and run `python -m scripts.wsgi_self_test`. Set the PythonAnywhere WSGI configuration file to import `application` from `app.wsgi` after adding this repository root to `sys.path` and activating the virtual environment. The free Beginner plan has limited disk and CPU resources and requires periodic renewal of its web app.

## Model and reproducibility

The predictor names and units are listed in [`docs/FEATURES.md`](docs/FEATURES.md). The model implementation and validation details are in [`docs/METHODS.md`](docs/METHODS.md). `scripts.self_test` uses synthetic measurements and checks prediction stability and SHAP additivity.

The included frozen model is required for prediction. Model reconstruction and cohort-level verification require the restricted project data; patient-level records are not included in this repository. Load the model file only from a trusted source because `joblib` uses Python pickle serialization.

