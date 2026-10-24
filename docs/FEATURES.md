# Clinical predictor set

The frozen random-forest model has **31 original clinical predictors**. Input names below are the exact names used by the model.

## Clinical and blood indicators selected by LASSO (17)

`Uric_acid`, `BUN`, `HbA1c`, `SBP`, `Hypertension`, `Age`, `Diabetes_duration_mo`, `HDL_C`, `Stroke_history`, `Albumin`, `TG`, `LDL_C`, `MPV`, `Glucose_0min`, `AST`, `CHD_history`, `Heart_rate`.

## Urine indicators (6)

`Urine_PRO`, `Urine_GLU`, `Urine_LEU`, `Urine_ERY`, `Urine_NIT`, `Urine_KET`.

These six fields use 0 for a negative result and 1 for a positive result in the web interface.

## TyG-related metabolic indicators (8)

`TyG`, `TyG-BMI`, `TyG-WC`, `TyG-WHR`, `TyG-WHtR`, `TyG-ABSI`, `TyG-BRI`, `TyG-WWI`.

The application calculates these indices from the entered anthropometric measurements, triglycerides, and fasting glucose. Triglycerides and glucose are entered in mmol/L. The TyG calculation converts them to mg/dL with factors 88.57 and 18.0182, respectively:

`TyG = ln[(TG_mg/dL × fasting_glucose_mg/dL) / 2]`.

Each composite is TyG multiplied by its named body-size index. The implementation is in [`app/predictor.py`](../app/predictor.py), function `derive`.

The interface requests raw measurements needed to calculate these derived indices, so the number of entry fields differs from the number of modeled predictors.
