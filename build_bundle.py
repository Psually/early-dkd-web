"""Freeze the published 31-feature/24-flag RF and 10 fitted MICE imputers.

Run once from the repository root. This never uses the validation or external
cohort to fit any model component. It verifies both cohorts' saved predictions.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

from runtime import ROOT, prepare_sklearn
prepare_sklearn()

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import IterativeImputer


HERE = Path(__file__).resolve().parent
WORK = ROOT / "work"
AUDIT = json.loads((ROOT / "outputs/dkd_fixed31_tyg_12model_roc/Fig2_fixed31_TyG_12model_audit.json").read_text(encoding="utf-8"))
MICE = json.loads((ROOT / "outputs/dkd_with_urine/DKD_panel_MICE_audit.json").read_text(encoding="utf-8"))
SPLIT = json.loads((WORK / "dkd_figure5a_split_ids.json").read_text(encoding="utf-8"))
FEATURES = AUDIT["all_original_predictors_each_model"]
FLAGS = AUDIT["original_missing_flags_each_model"]
CONTINUOUS = MICE["continuous_inputs"]
CATEGORICAL = MICE["categorical_inputs"]
RAW = CONTINUOUS + CATEGORICAL
MODES = MICE["categorical_training_modes"]
assert len(FEATURES) == 31 and len(FLAGS) == 24 and len(RAW) == 30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path, urine_path: Path) -> pd.DataFrame:
    rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))).set_index("study_id")
    urine = pd.read_pickle(urine_path)
    return rows.join(urine[["Urine_PRO", "Urine_GLU", "Urine_LEU", "Urine_ERY", "Urine_NIT", "Urine_KET"]].loc[rows.index])


def source_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows[RAW].astype(float).copy()
    for name, mode in MODES.items():
        x[name] = x[name].fillna(mode)
    return x


def feature_matrix(completed: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    x = completed.loc[observed.index, FEATURES].astype(float).copy()
    for name in FLAGS:
        x[f"missing_{name}"] = observed[name].isna().to_numpy(float)
    assert x.columns.to_list() == FEATURES + [f"missing_{name}" for name in FLAGS]
    assert np.isfinite(x.to_numpy()).all()
    return x


def main() -> None:
    dev = load_raw(WORK / "psm_clean_baseline_values.json", WORK / "dkd_urine_feature_matrix_1243.pkl")
    ext = load_raw(WORK / "third_center_external_baseline.json", WORK / "shenyang_urine6_1345.pkl")
    train = dev.loc[SPLIT["train_ids"]]
    valid = dev.loc[SPLIT["test_ids"]]
    assert (len(dev), len(train), len(valid), len(ext)) == (1243, 870, 373, 1345)
    train_x, dev_x, ext_x = (source_matrix(x) for x in (train, dev, ext))
    minimum = np.asarray([float(train_x[c].min()) if c in CONTINUOUS else 0 for c in RAW])
    maximum = np.asarray([max(float(train_x[c].max()), minimum[i] + 1) for i, c in enumerate(RAW)])
    threshold = float(pd.read_csv(ROOT / "outputs/dkd_fixed31_tyg_12model_roc/Fig2_fixed31_TyG_PR_AUC_confusion_metrics.csv").set_index("model").loc["Random forest", "train_oof_Youden_threshold"])
    assert np.isclose(threshold, 0.4180193107)
    models, imputers, valid_probs, ext_probs = [], [], [], []
    manifest = {(r["cohort"], r["draw"]): r for r in MICE["files"]}
    for draw in range(1, 11):
        imputer = IterativeImputer(max_iter=10, sample_posterior=True,
                                   initial_strategy="median", random_state=20261022 + draw,
                                   min_value=minimum, max_value=maximum, skip_complete=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            imputer.fit(train_x)
            # Advance posterior RNG in the exact order of the original pipeline.
            replay_dev = pd.DataFrame(imputer.transform(dev_x), index=dev.index, columns=RAW)
            replay_ext = pd.DataFrame(imputer.transform(ext_x), index=ext.index, columns=RAW)
        for name in CATEGORICAL:
            replay_dev[name], replay_ext[name] = dev_x[name], ext_x[name]
        for cohort, replay in (("development", replay_dev), ("external", replay_ext)):
            saved = pd.read_pickle(ROOT / manifest[(cohort, draw)]["path"])
            for name in set(FEATURES) & set(RAW):
                error = float(np.max(np.abs(saved[name].to_numpy(float) - replay[name].to_numpy(float))))
                assert error < 1e-9, (cohort, draw, name, error)
        zd = pd.read_pickle(ROOT / manifest[("development", draw)]["path"])
        ze = pd.read_pickle(ROOT / manifest[("external", draw)]["path"])
        model = RandomForestClassifier(n_estimators=180, min_samples_leaf=10,
                                       max_features="sqrt", n_jobs=1, random_state=20261022)
        model.fit(feature_matrix(zd, train), train.y.to_numpy(int))
        valid_probs.append(model.predict_proba(feature_matrix(zd, valid))[:, 1])
        ext_probs.append(model.predict_proba(feature_matrix(ze, ext))[:, 1])
        models.append(model)
        imputers.append(imputer)
        print(f"Reconstructed imputation and RF {draw}/10", flush=True)
    frozen_valid = pd.read_csv(WORK / "dkd_fixed31_tyg_12model_validation_predictions_deidentified.csv").set_index("study_id").loc[valid.index]
    frozen_ext = pd.read_csv(WORK / "dkd_fixed31_tyg_fig9_external_predictions_deidentified.csv").set_index("study_id").loc[ext.index]
    internal_error = float(np.max(np.abs(np.mean(valid_probs, axis=0) - frozen_valid["Random forest"].to_numpy(float))))
    external_col = next(c for c in frozen_ext.columns if c not in ("outcome", "center", "label_support") and pd.api.types.is_numeric_dtype(frozen_ext[c]))
    external_error = float(np.max(np.abs(np.mean(ext_probs, axis=0) - frozen_ext[external_col].to_numpy(float))))
    assert internal_error < 1e-9 and external_error < 1e-9, (internal_error, external_error, external_col)
    bundle = {"version": "fixed31_TyG_2026-10-24", "features": FEATURES, "flags": FLAGS,
              "continuous": CONTINUOUS, "categorical": CATEGORICAL, "raw_inputs": RAW,
              "modes": MODES, "min": minimum, "max": maximum, "threshold": threshold,
              "models": models, "imputers": imputers}
    artifact = HERE / "frozen_rf31_bundle.joblib"
    joblib.dump(bundle, artifact, compress=3)
    report = {"bundle_sha256": digest(artifact), "internal_replay_max_abs_error": internal_error,
              "external_replay_max_abs_error": external_error, "external_probability_column": external_col,
              "training_n": 870, "internal_validation_n": 373, "external_validation_n": 1345,
              "original_predictors": 31, "missingness_indicators": 24, "imputation_draws": 10,
              "threshold": threshold,
              "source_sha256": {"model_audit": digest(ROOT / "outputs/dkd_fixed31_tyg_12model_roc/Fig2_fixed31_TyG_12model_audit.json"),
                                "mice_audit": digest(ROOT / "outputs/dkd_with_urine/DKD_panel_MICE_audit.json")}}
    (HERE / "bundle_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
