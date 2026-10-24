# Model and validation notes

## Outcome and cohorts

The frozen application predicts the study-defined, contemporaneous early kidney disease grouping among adults with obesity and type 2 diabetes. It is a case-identification model rather than a prospective risk model or a standalone CKD diagnosis. The original pre-PSM development cohort had 1,243 patients: 870 for training and 373 for internal holdout validation. A separate cohort of 1,345 patients was used for external validation.

Urine protein may proxy part of the ACR-based outcome definition. The historical LASSO feature selection used the full development cohort before the 870/373 split. These points should be disclosed when interpreting discrimination and external validation.

## Predictors and preprocessing

Each model uses the same 31 original clinical predictors listed in [`FEATURES.md`](FEATURES.md). The frozen implementation also retains 24 original-missingness indicator columns as auxiliary inputs. These are derived from whether selected raw measurements were missing at entry; they are not additional clinical measurements. The model matrix therefore has 55 numerical columns per imputation draw.

Missing continuous values use ten training-fitted chained-equation imputation draws; categorical values use training modes. The final probability is the mean of ten frozen random-forest probabilities. Each forest has 180 trees, minimum leaf size 10, and `max_features='sqrt'`. The seed recorded for model construction is 20261022.

The decision threshold is 0.4180193107, selected from training out-of-fold Youden J. Thresholded labels depend on this threshold; the probability remains the primary output.

## Individual explanation

The application computes Tree SHAP values for the frozen forests. It displays contributions for the 31 clinical predictors and groups the remaining auxiliary contributions into `Other`. The top-ten waterfall plus `Other` is checked for additive agreement with the reported probability.

## Reproducibility files

- [`scripts/build_bundle.py`](../scripts/build_bundle.py) reconstructs the frozen bundle from restricted, local project files.
- [`scripts/verify_web_model.py`](../scripts/verify_web_model.py) compares predictions with the saved external cohort predictions when the restricted data are available.
- [`scripts/self_test.py`](../scripts/self_test.py) uses synthetic measurements and needs no patient records.
- [`bundle_audit.json`](bundle_audit.json) and [`web_verification.json`](web_verification.json) preserve the original reconstruction and verification summaries.
- [`../models/MODEL_SHA256.txt`](../models/MODEL_SHA256.txt) records the frozen model hash; the application verifies it before loading the model.

To run the restricted-data scripts, set `DKD_PROJECT_ROOT` to the directory containing the expected `work/` and `outputs/` files, then run them from this repository root as Python modules. Patient-level files and cohort prediction rows are intentionally excluded from this upload package.
