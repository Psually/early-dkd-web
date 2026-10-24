# Early DKD Prediction Web App — v1.0.0

A local, English-language research interface for the frozen pre-PSM random-forest model in adults with obesity and type 2 diabetes. It accepts 30 raw fields, calculates eight TyG-related indices, and displays an individual SHAP waterfall. Only the 31 original clinical predictor contributions are displayed; the remaining model inputs are aggregated into **Other**. The model uses the original 24 missingness indicators internally and has **not** been refitted for this release.

This private repository includes the approved frozen model bundle, so it can run after installing the listed dependencies. It contains no patient-level source data.

## Which archive to use

- `early-dkd-web-v1.0.0-source.zip` is ready for a public source repository and contains no fitted model or patient records. Obtain the approved model bundle separately and place `frozen_rf31_bundle.joblib` in this directory before running.
- `early-dkd-web-v1.0.0-complete.zip` includes the frozen fitted model for controlled distribution. Review institutional approval before making this binary public. It contains no raw patient rows.

Both archives contain the same source code. `MODEL_SHA256.txt` identifies the exact approved model; startup verifies its hash before deserializing it. The full archive and this private repository include the model.

## Run locally

Use Python 3.12. Create an environment and install the locked direct dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe self_test.py
.\.venv\Scripts\python.exe serve.py
```

Open <http://127.0.0.1:8765/>. On Windows, `start_web.cmd` uses `.venv` if present. The server binds only to `127.0.0.1` and does not store submitted patient inputs or results. Load only a model bundle from a trusted source: `joblib` files use Python pickle serialization.

## Fixed model contract

- Original clinical predictors: 17 historical LASSO indicators + six urine indicators + eight TyG-family indices = **31**.
- Auxiliary missingness indicators: **24**, processed internally in every model. There are 55 numerical model inputs per imputation draw.
- Missing data: ten training-fitted Bayesian chained-equation imputers; categorical training modes.
- Algorithm: mean probability from ten frozen random forests; each has 180 trees, minimum leaf size 10, `max_features='sqrt'`, seed 20261022.
- Decision threshold: **0.4180193107**, selected from training out-of-fold Youden J.
- Development training / internal holdout / independent external cohort: **870 / 373 / 1,345**.

The fitted model was reconstructed against both saved cohorts before freezing. The internal maximum absolute probability difference was below 1e-15; the external comparison to saved ten-decimal predictions was below 5e-11. `bundle_audit.json` and `web_verification.json` record these checks. `self_test.py` uses only synthetic measurements and checks the fixed probability, repeatability, SHAP additivity, 31-feature display, and waterfall generation.

## Rebuilding from restricted data

`build_bundle.py` and `verify_web_model.py` are included for authorized reviewers with access to the private project data. They require the existing `work/` and `outputs/` inputs and are not required to serve the frozen model. Set `DKD_PROJECT_ROOT` to the local project root before running them. No patient-level source data, identifiers, workbooks, or cohort prediction rows are included in either archive.

## Research scope

This is a contemporaneous case-identification model, not a standalone CKD diagnosis. The urine protein input may proxy the ACR-based endpoint. The historical LASSO selection used the entire 1,243-person development cohort; this limitation should accompany validation claims. The UI structure was inspired by Fig. 10 of the reference paper; the public reference repository did not provide the web source code, and this implementation is original.

No software license is assigned in this package; the repository owner should choose one before publication.
