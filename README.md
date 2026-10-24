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

## Model and reproducibility

The predictor names and units are listed in [`docs/FEATURES.md`](docs/FEATURES.md). The model implementation and validation details are in [`docs/METHODS.md`](docs/METHODS.md). `scripts.self_test` uses synthetic measurements and checks prediction stability and SHAP additivity.

The included frozen model is required for prediction. Model reconstruction and cohort-level verification require the restricted project data; patient-level records are not included in this repository. Load the model file only from a trusted source because `joblib` uses Python pickle serialization.

## Reference

Repository organization was informed by the [early-stage CKD prediction example](https://github.com/Jiayi-Njucm/early-stage-ckd-prediction). This application's implementation and fitted model are separate from that example.
