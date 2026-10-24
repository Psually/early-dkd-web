"""Single-patient inference for the frozen pre-PSM early-DKD RF ensemble."""
from __future__ import annotations

import base64
import hashlib
import copy
import io
import math
import threading
import warnings
from pathlib import Path

from runtime import prepare_sklearn
prepare_sklearn()

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


HERE = Path(__file__).resolve().parent
URINE = ["Urine_PRO", "Urine_GLU", "Urine_LEU", "Urine_ERY", "Urine_NIT", "Urine_KET"]
LABELS = {
    "Age": ("Age", "years"), "Sex": ("Sex", ""), "Height": ("Height", "cm"),
    "Weight": ("Weight", "kg"), "BMI": ("BMI", "kg/m²"),
    "Waist_circ": ("Waist circumference", "cm"), "Hip_circ": ("Hip circumference", "cm"),
    "Diabetes_duration_mo": ("Diabetes duration", "months"),
    "Hypertension": ("Hypertension history", ""), "Stroke_history": ("Stroke history", ""),
    "CHD_history": ("Coronary heart disease history", ""), "SBP": ("Systolic blood pressure", "mmHg"),
    "DBP": ("Diastolic blood pressure", "mmHg"), "Heart_rate": ("Heart rate", "beats/min"),
    "Uric_acid": ("Uric acid", "μmol/L"), "BUN": ("Blood urea nitrogen", "mmol/L"),
    "HbA1c": ("HbA1c", "%"), "HDL_C": ("HDL-C", "mmol/L"),
    "Albumin": ("Albumin", "g/L"), "TG": ("Triglycerides", "mmol/L"),
    "LDL_C": ("LDL-C", "mmol/L"), "MPV": ("Mean platelet volume", "fL"),
    "Glucose_0min": ("Fasting plasma glucose", "mmol/L"), "AST": ("AST", "U/L"),
    "Urine_PRO": ("Urine protein", ""), "Urine_GLU": ("Urine glucose", ""),
    "Urine_LEU": ("Urine leukocytes", ""), "Urine_ERY": ("Urine erythrocytes", ""),
    "Urine_NIT": ("Urine nitrite", ""), "Urine_KET": ("Urine ketones", ""),
}
GROUPS = [
    ("basic", "Basic Information", ["Age", "Sex", "Height", "Weight", "BMI", "Waist_circ", "Hip_circ", "Diabetes_duration_mo", "Hypertension", "Stroke_history", "CHD_history", "SBP", "DBP", "Heart_rate"]),
    ("blood", "Blood Tests", ["Uric_acid", "BUN", "HbA1c", "HDL_C", "Albumin", "TG", "LDL_C", "MPV", "Glucose_0min", "AST"]),
    ("urine", "Urine Tests", URINE),
]
BOUNDS = {
    "Age": (18, 110), "Height": (120, 210), "Weight": (40, 250),
    "BMI": (28, 80), "Waist_circ": (60, 180), "Hip_circ": (70, 180),
    "Diabetes_duration_mo": (0, 720), "SBP": (70, 300), "DBP": (30, 150),
    "Heart_rate": (30, 180), "Uric_acid": (80, 900), "BUN": (1, 40),
    "HbA1c": (4, 22), "HDL_C": (.2, 4), "Albumin": (20, 60),
    "TG": (.2, 30), "LDL_C": (.2, 12), "MPV": (4, 20),
    "Glucose_0min": (2, 40), "AST": (3, 1000),
}


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Identical formulas and units to work/impute_dkd_feature_panels.py."""
    out = df.copy()
    h, w, wt, bmi, hip = (out[c].astype(float) for c in ("Height", "Waist_circ", "Weight", "BMI", "Hip_circ"))
    tg, glu = out.TG.astype(float), out.Glucose_0min.astype(float)
    whr = w / hip
    shape = 1 - (w / (np.pi * h)) ** 2
    out["WWI"] = w / np.sqrt(wt)
    out["ABSI"] = (w / 100) / (bmi ** (2 / 3) * (h / 100) ** .5)
    out["WHtR"] = w / h
    out["BRI"] = 364.2 - 365.5 * np.sqrt(shape.where(shape >= 0))
    out["TyG"] = np.log(tg * 88.57 * glu * 18.0182 / 2)
    for name, factor in {"TyG-BMI": bmi, "TyG-WC": w, "TyG-WHR": whr,
                         "TyG-WHtR": out.WHtR, "TyG-ABSI": out.ABSI,
                         "TyG-BRI": out.BRI, "TyG-WWI": out.WWI}.items():
        out[name] = out.TyG * factor
    return out.replace([np.inf, -np.inf], np.nan)


class InputError(ValueError):
    pass


class Predictor:
    def __init__(self, artifact: Path = HERE / "frozen_rf31_bundle.joblib"):
        if not artifact.is_file():
            raise FileNotFoundError("Place the approved frozen_rf31_bundle.joblib beside predictor.py")
        expected = (HERE / "MODEL_SHA256.txt").read_text(encoding="ascii").strip()
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Frozen model checksum mismatch")
        self.bundle = joblib.load(artifact)
        self.features = self.bundle["features"]
        self.flags = self.bundle["flags"]
        self.all_features = self.features + [f"missing_{name}" for name in self.flags]
        assert len(self.features) == 31 and len(self.all_features) == 55
        self.explainers = [shap.TreeExplainer(model, model_output="raw",
                                            feature_perturbation="tree_path_dependent")
                           for model in self.bundle["models"]]
        self.plot_lock = threading.Lock()

    def schema(self) -> dict:
        groups = []
        for group_id, title, names in GROUPS:
            fields = []
            for name in names:
                label, unit = LABELS[name]
                field = {"name": name, "label": label, "unit": unit,
                         "kind": "sex" if name == "Sex" else "binary" if name in self.bundle["categorical"] else "number",
                         "required": name in ("Age", "Sex")}
                if name in BOUNDS:
                    field["min"], field["max"] = BOUNDS[name]
                fields.append(field)
            groups.append({"id": group_id, "title": title, "fields": fields})
        return {"groups": groups, "model": "RF · 31 clinical predictors",
                "threshold": self.bundle["threshold"]}

    def parse(self, values: dict) -> pd.DataFrame:
        if not isinstance(values, dict):
            raise InputError("Please submit a JSON object.")
        unknown = set(values) - set(self.bundle["raw_inputs"])
        if unknown:
            raise InputError("The request contains an undefined predictor.")
        row = {}
        for name in self.bundle["raw_inputs"]:
            value = values.get(name)
            if value is None or value == "":
                row[name] = np.nan
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise InputError(f"{LABELS[name][0]} must be numeric.") from None
            if not math.isfinite(number):
                raise InputError(f"{LABELS[name][0]} must be a finite number.")
            if name in self.bundle["categorical"] and number not in (0, 1):
                raise InputError(f"{LABELS[name][0]} accepts only Negative/No (0) or Positive/Yes (1).")
            if name in BOUNDS and not BOUNDS[name][0] <= number <= BOUNDS[name][1]:
                raise InputError(f"{LABELS[name][0]} is outside the entry range {BOUNDS[name][0]}–{BOUNDS[name][1]}.")
            row[name] = number
        if np.isnan(row["Age"]) or np.isnan(row["Sex"]):
            raise InputError("Please enter age and sex.")
        if sum(np.isfinite(row[n]) for n in self.features if n in row) < 4:
            raise InputError("Too few measurements are available. Please add clinical, blood, or urine results.")
        return pd.DataFrame([row], columns=self.bundle["raw_inputs"])

    def _matrix(self, imputed: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
        derived = derive(imputed)
        x = derived[self.features].copy()
        for name in self.flags:
            x[f"missing_{name}"] = float(observed[name].isna().iloc[0])
        x = x[self.all_features]
        if not np.isfinite(x.to_numpy(float)).all():
            raise InputError("TyG indices cannot be calculated. Check height, weight, waist, hip, triglycerides, and glucose.")
        return x

    def predict(self, values: dict, *, plot: bool = True) -> dict:
        observed = self.parse(values)
        source = observed.copy()
        for name, mode in self.bundle["modes"].items():
            source[name] = source[name].fillna(mode)
        probabilities, bases, contributions, completed = [], [], [], []
        for imputer, model, explainer in zip(self.bundle["imputers"], self.bundle["models"], self.explainers):
            # IterativeImputer's posterior RNG is stateful: clone it so the same
            # input always yields the same score, irrespective of request order.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                data = copy.deepcopy(imputer).transform(source)
            num = pd.DataFrame(data, columns=self.bundle["raw_inputs"])
            for name in self.bundle["categorical"]:
                num[name] = source[name]
            x = self._matrix(num, observed)
            p = float(model.predict_proba(x)[0, 1])
            result = explainer.shap_values(x, check_additivity=False)
            phi = np.asarray(result[1] if isinstance(result, list) else result, dtype=float)
            if phi.shape == (1, 55, 2):
                phi = phi[:, :, 1]
            phi = phi.reshape(55)
            expected = np.asarray(explainer.expected_value).reshape(-1)
            base = float(expected[1] if len(expected) == 2 else expected[0])
            assert abs(base + float(phi.sum()) - p) < 1e-9
            probabilities.append(p)
            bases.append(base)
            contributions.append(phi)
            completed.append(x.iloc[0].to_numpy(float))
        probability = float(np.mean(probabilities))
        baseline = float(np.mean(bases))
        phi = np.mean(contributions, axis=0)
        mean_values = np.mean(completed, axis=0)
        assert abs(baseline + float(phi.sum()) - probability) < 1e-9
        clinical_order = np.argsort(-np.abs(phi[:len(self.features)]), kind="stable")
        top = clinical_order[:10]
        other_indices = np.array([i for i in range(len(self.all_features)) if i not in set(top)])
        other = float(phi[other_indices].sum())
        clinical = [{"name": self.features[i], "value": float(mean_values[i]),
                     "shap_probability": float(phi[i])} for i in clinical_order]
        assert len(clinical) == 31 and len(other_indices) == 45
        assert abs(baseline + float(phi[top].sum()) + other - probability) < 1e-9
        result = {"probability": probability, "percent": round(100 * probability, 1),
                  "threshold": self.bundle["threshold"],
                  "classification": "Above threshold" if probability >= self.bundle["threshold"] else "Below threshold",
                  "baseline": baseline, "missing_count": int(observed.isna().sum(axis=1).iloc[0]),
                  "top_contributions": clinical[:10], "all_contributions": clinical,
                  "other_model_inputs_contribution": other,
                  "tyg_indices": {name: float(mean_values[self.all_features.index(name)])
                                  for name in self.features if name.startswith("TyG")}}
        if plot:
            labels = [f"{mean_values[i]:.3g} = {self.all_features[i]}" for i in top]
            labels.append("Other model inputs")
            explanation = shap.Explanation(values=np.r_[phi[top], other],
                                           base_values=baseline, feature_names=labels)
            with self.plot_lock:
                plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                                     "figure.facecolor": "white", "savefig.facecolor": "white"})
                shap.plots.waterfall(explanation, max_display=11, show=False)
                fig = plt.gcf()
                fig.set_size_inches(9, 6.4)
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight", pad_inches=.16)
                plt.close(fig)
            result["waterfall_png"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        return result
