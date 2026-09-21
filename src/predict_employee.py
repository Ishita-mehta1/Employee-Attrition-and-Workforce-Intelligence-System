import argparse
import os
import sys
import joblib
import pandas as pd

# Allow this file to be executed directly from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import MODEL_FEATURES, local_sensitivity, risk_label

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "employee_data.csv")
PRED_PATH = os.path.join(BASE_DIR, "outputs", "predictions", "employee_predictions.csv")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")


def load_employee(employee_id=None, name=None):
    df = pd.read_csv(PRED_PATH if os.path.exists(PRED_PATH) else DATA_PATH)

    if employee_id:
        matches = df[df["EmployeeID"].astype(str).str.lower() == employee_id.lower()]
    elif name:
        matches = df[df["Name"].astype(str).str.lower() == name.lower()]
    else:
        raise ValueError("Provide --employee-id or --name.")

    if matches.empty:
        raise ValueError("Employee not found.")
    return matches.iloc[0]


def analyze(employee_id=None, name=None):
    employee = load_employee(employee_id, name)
    row = employee[MODEL_FEATURES].to_dict()
    row_df = pd.DataFrame([row])

    attrition = joblib.load(os.path.join(MODEL_DIR, "attrition_model.joblib"))
    promotion = joblib.load(os.path.join(MODEL_DIR, "promotion_model.joblib"))
    training = joblib.load(os.path.join(MODEL_DIR, "training_need_model.joblib"))
    performance = joblib.load(os.path.join(MODEL_DIR, "performance_model.joblib"))

    attrition_prob = float(attrition.predict_proba(row_df)[0, 1] * 100)
    promotion_prob = float(promotion.predict_proba(row_df)[0, 1] * 100)
    training_prob = float(training.predict_proba(row_df)[0, 1] * 100)
    predicted_performance = float(performance.predict(row_df)[0])

    # The model-based risk score is intentionally transparent.
    risk_score = (
        0.45 * attrition_prob
        + 0.20 * (100 - employee["PerformanceScore"] * 20)
        + 0.15 * (100 - employee["JobSatisfaction"] * 20)
        + 0.10 * (100 - employee["WorkLifeBalance"] * 20)
        + 0.10 * training_prob
    )
    risk_score = max(0, min(100, risk_score))

    explanation = local_sensitivity(
        attrition, row, MODEL_FEATURES
    )

    print("\n" + "=" * 60)
    print("INDIVIDUAL EMPLOYEE INTELLIGENCE")
    print("=" * 60)
    print(f"Employee ID       : {employee['EmployeeID']}")
    print(f"Name              : {employee['Name']}")
    print(f"Department        : {employee['Department']}")
    print(f"Job Role          : {employee['JobRole']}")
    print(f"Age               : {employee['Age']}")
    print(f"Years at Company  : {employee['YearsAtCompany']}")
    print(f"Job Satisfaction  : {employee['JobSatisfaction']}/5")
    print(f"Performance       : {employee['PerformanceScore']}/5")
    print(f"Training Hours    : {employee['TrainingHours']}")
    print("-" * 60)
    print(f"Attrition Risk    : {attrition_prob:.2f}%")
    print(f"Promotion Score   : {promotion_prob:.2f}%")
    print(f"Training Need     : {training_prob:.2f}%")
    print(f"Predicted Perf.   : {predicted_performance:.2f}/5")
    print(f"Employee Risk     : {risk_score:.2f}/100 ({risk_label(risk_score)})")
    print(f"Anomaly           : {'Yes' if employee.get('Anomaly', 0) == 1 else 'No'}")
    print(f"Employee Segment  : {employee.get('EmployeeSegment', 'N/A')}")
    print("-" * 60)

    print("EXPLAINABLE AI - IMPORTANT LOCAL FACTORS")
    if explanation.empty:
        print("Explanation unavailable.")
    else:
        for _, item in explanation.iterrows():
            direction = "increases" if item["ChangeInProbability"] > 0 else "decreases"
            print(
                f"- {item['Feature']}: changing it to a reference value "
                f"{direction} predicted attrition probability by "
                f"{abs(item['ChangeInProbability']) * 100:.2f} percentage points."
            )
    print("=" * 60)

    return {
        "EmployeeID": employee["EmployeeID"],
        "Name": employee["Name"],
        "AttritionProbability": round(attrition_prob, 2),
        "PromotionProbability": round(promotion_prob, 2),
        "TrainingNeedProbability": round(training_prob, 2),
        "PredictedPerformance": round(predicted_performance, 2),
        "RiskScore": round(risk_score, 2),
        "RiskLevel": risk_label(risk_score),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--employee-id")
    parser.add_argument("--name")
    args = parser.parse_args()

    analyze(args.employee_id, args.name)
