import os
import json
import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction",
    "WorkLifeBalance",
    "PerformanceScore",
    "TrainingHours",
    "ProjectsCompleted",
]

CATEGORICAL_FEATURES = [
    "Department",
    "JobRole",
    "EducationLevel",
    "City",
    "Overtime",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_dirs():
    for folder in [
        "outputs/models",
        "outputs/charts",
        "outputs/predictions",
        "outputs/reports",
    ]:
        os.makedirs(folder, exist_ok=True)


def risk_label(score):
    if score < 35:
        return "Low"
    if score < 65:
        return "Medium"
    return "High"


def safe_json_dump(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def feature_importance_table(model, preprocessor):
    """Return readable global feature importance when supported."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    names = preprocessor.get_feature_names_out()
    values = model.feature_importances_
    result = pd.DataFrame({"Feature": names, "Importance": values})
    result["Feature"] = (
        result["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    return result.sort_values("Importance", ascending=False).head(15)


def local_sensitivity(model_pipeline, row, features, positive_class=True):
    """
    Simple model-agnostic local explanation:
    replace one feature with a representative alternative and measure
    the change in predicted probability. This avoids requiring SHAP.
    """
    base = row.copy()
    try:
        base_prob = float(model_pipeline.predict_proba(pd.DataFrame([base]))[0, 1])
    except Exception:
        return pd.DataFrame(columns=["Feature", "ChangeInProbability"])

    records = []
    numeric_medians = {
        "Age": 35, "MonthlyIncome": 50000, "YearsAtCompany": 4,
        "JobSatisfaction": 3, "WorkLifeBalance": 3,
        "PerformanceScore": 3.5, "TrainingHours": 30, "ProjectsCompleted": 5
    }

    alternatives = {
        "Age": numeric_medians["Age"],
        "MonthlyIncome": numeric_medians["MonthlyIncome"],
        "YearsAtCompany": numeric_medians["YearsAtCompany"],
        "JobSatisfaction": 3,
        "WorkLifeBalance": 3,
        "PerformanceScore": 3.5,
        "TrainingHours": 30,
        "ProjectsCompleted": 5,
        "Department": "IT",
        "JobRole": "Analyst",
        "EducationLevel": "Bachelor",
        "City": "Delhi",
        "Overtime": "No",
    }

    for feature in features:
        if feature not in base or feature not in alternatives:
            continue
        changed = base.copy()
        changed[feature] = alternatives[feature]
        try:
            new_prob = float(
                model_pipeline.predict_proba(pd.DataFrame([changed]))[0, 1]
            )
            records.append({
                "Feature": feature,
                "ChangeInProbability": round(new_prob - base_prob, 4)
            })
        except Exception:
            pass

    return (
        pd.DataFrame(records)
        .sort_values("ChangeInProbability", key=lambda s: s.abs(), ascending=False)
        .head(8)
    )
