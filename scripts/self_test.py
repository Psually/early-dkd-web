"""Synthetic, patient-free release gate for the frozen inference bundle."""
from __future__ import annotations
import math
from app.predictor import InputError, Predictor

SYNTHETIC = {"Age":60,"Sex":1,"Height":170,"Weight":92,"BMI":31.8,
             "Waist_circ":105,"Hip_circ":108,"TG":2.1,"Glucose_0min":8.3,
             "Urine_PRO":1}
EXPECTED = 0.1297303301152633

def main() -> None:
    model = Predictor()
    one = model.predict(SYNTHETIC)
    two = model.predict(SYNTHETIC, plot=False)
    assert math.isclose(one["probability"], EXPECTED, abs_tol=1e-8)
    assert one["probability"] == two["probability"]
    assert len(one["all_contributions"]) == 31
    assert len(one["top_contributions"]) == 10
    assert all(not row["name"].startswith("missing_") for row in one["all_contributions"])
    additive = one["baseline"] + sum(row["shap_probability"] for row in one["top_contributions"]) + one["other_model_inputs_contribution"]
    assert abs(additive - one["probability"]) < 1e-9
    assert one["waterfall_png"].startswith("data:image/png;base64,")
    try:
        model.predict({**SYNTHETIC, "Age": None}, plot=False)
    except InputError as error:
        assert str(error) == "Please enter age and sex."
    else:
        raise AssertionError("Missing age was accepted")
    print("PASS: frozen probability, repeatability, 31-feature display, SHAP additivity, waterfall, validation")

if __name__ == "__main__":
    main()
