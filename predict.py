"""Score one loan applicant with the saved v3 model."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"
MODEL = joblib.load(MODEL_DIR / "risk_model_v3.joblib")
META = json.loads((MODEL_DIR / "model_meta.json").read_text())
FEATURES = META["features"]


def encode(applicant: dict) -> pd.DataFrame:
    """Turn plain answers into the exact 25 columns the model was trained on."""
    needed = list(META["numeric"]) + list(META["categorical"])
    missing = [k for k in needed if k not in applicant]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")

    for column, info in META["categorical"].items():
        if applicant[column] not in info["options"]:
            raise ValueError(f"{column}: '{applicant[column]}' is not one of {info['options']}")

    row = {}
    for feature in FEATURES:
        if feature in META["numeric"]:
            row[feature] = float(applicant[feature])
        else:                                        # a 0/1 column, e.g. "EmploymentStatus_Unemployed"
            column, value = feature.split("_", 1)
            row[feature] = int(applicant[column] == value)
    return pd.DataFrame([row], columns=FEATURES)     # same names, same order as training


def predict(applicant: dict) -> float:
    """Predicted RiskScore, kept inside the range seen in training."""
    raw = float(MODEL.predict(encode(applicant))[0])
    return float(np.clip(raw, META["target"]["min"], META["target"]["max"]))


def check_inputs(applicant: dict) -> list:
    """Warnings for numbers outside the range the model learned from."""
    notes = []
    for name, m in META["numeric"].items():
        value = float(applicant[name])
        if not m["min"] <= value <= m["max"]:
            notes.append(
                f"{name} = {value:,.10g} is outside the training range "
                f"({m['min']:,.10g} to {m['max']:,.10g}), so treat the score with care."
            )
    return notes


def percentile_rank(score: float) -> float:
    """Share of training borrowers (0 to 100) with a lower score."""
    return float(np.interp(score, META["target"]["percentiles"], range(101)))


def explain(applicant: dict, top: int = 5) -> pd.Series:
    """Points each feature adds (+) or removes (-) versus the average training borrower."""
    x = encode(applicant).iloc[0]
    effect = (x - pd.Series(META["means"])) * pd.Series(MODEL.coef_, index=FEATURES)
    return effect.sort_values(key=abs, ascending=False).head(top)


def risk_band(score: float) -> str:
    """Low / Medium / High, by where the score falls among training borrowers."""
    cuts = META["target"]["percentiles"]
    if score < cuts[33]:
        return "Low"
    if score < cuts[67]:
        return "Medium"
    return "High"