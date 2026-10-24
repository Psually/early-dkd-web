"""Reproducibility and endpoint smoke tests; no patient rows are exported."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import HERE, InputError, Predictor
from runtime import ROOT


def main() -> None:
    model = Predictor()
    raw = pd.DataFrame(json.loads((ROOT / "work/third_center_external_baseline.json").read_text(encoding="utf-8"))).set_index("study_id")
    urine = pd.read_pickle(ROOT / "work/shenyang_urine6_1345.pkl")
    raw = raw.join(urine[model.bundle["categorical"][-6:]].loc[raw.index])
    complete = raw[model.bundle["raw_inputs"]].notna().all(axis=1)
    assert int(complete.sum()) > 0
    case_id = complete[complete].index[0]
    patient = raw.loc[case_id, model.bundle["raw_inputs"]].to_dict()
    patient = {name: float(value) for name, value in patient.items()}
    expected = pd.read_csv(ROOT / "work/dkd_fixed31_tyg_fig9_external_predictions_deidentified.csv").set_index("study_id").loc[case_id, "predicted_probability"]
    result = model.predict(patient)
    repeat = model.predict(patient, plot=False)
    assert abs(result["probability"] - expected) < 5e-11, (result["probability"], expected)
    assert abs(result["probability"] - repeat["probability"]) < 1e-15
    assert abs(result["baseline"] + sum(x["shap_probability"] for x in result["top_contributions"]) + result["other_model_inputs_contribution"] - result["probability"]) < 1e-9
    assert len(result["all_contributions"]) == 31 and len(result["top_contributions"]) == 10
    assert all(not x["name"].startswith("missing_") for x in result["all_contributions"])
    assert result["waterfall_png"].startswith("data:image/png;base64,")
    assert result["missing_count"] == 0
    sparse = {"Age": 60, "Sex": 1, "Height": 170, "Weight": 92,
              "BMI": 31.8, "Waist_circ": 105, "Hip_circ": 108,
              "TG": 2.1, "Glucose_0min": 8.3, "Urine_PRO": 1}
    sparse_first = model.predict(sparse, plot=False)
    sparse_second = model.predict(sparse, plot=False)
    assert sparse_first["missing_count"] == 20
    assert abs(sparse_first["probability"] - sparse_second["probability"]) < 1e-15
    assert abs(sparse_first["baseline"] + sum(x["shap_probability"] for x in sparse_first["top_contributions"]) + sparse_first["other_model_inputs_contribution"] - sparse_first["probability"]) < 1e-9
    try:
        model.predict({**patient, "Age": math.inf}, plot=False)
    except InputError:
        pass
    else:
        raise AssertionError("Non-finite age was accepted")
    report = {"complete_case_external_replay_abs_error": abs(result["probability"] - expected),
              "repeat_prediction_abs_error": abs(result["probability"] - repeat["probability"]),
              "repeat_prediction_with_missing_values_abs_error": abs(sparse_first["probability"] - sparse_second["probability"]),
              "shap_additivity_abs_error": abs(result["baseline"] + sum(x["shap_probability"] for x in result["top_contributions"]) + result["other_model_inputs_contribution"] - result["probability"]),
              "waterfall_png_bytes_approx": len(result["waterfall_png"]) * 3 // 4,
              "tested_model_inputs": 55, "displayed_clinical_contributions": 31,
              "external_complete_cases_available": int(complete.sum())}
    (HERE / "web_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
