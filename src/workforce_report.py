import os
import pandas as pd

from src.utils import risk_label, safe_json_dump

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_PATH = os.path.join(BASE_DIR, "outputs", "predictions", "employee_predictions.csv")


def create_workforce_report():
    df = pd.read_csv(PRED_PATH)

    report = {
        "total_employees": int(len(df)),
        "high_attrition_risk_count": int((df["AttritionProbability"] >= 65).sum()),
        "medium_attrition_risk_count": int(
            ((df["AttritionProbability"] >= 35) &
             (df["AttritionProbability"] < 65)).sum()
        ),
        "promotion_suitable_count": int((df["PromotionProbability"] >= 60).sum()),
        "training_need_count": int((df["TrainingNeedProbability"] >= 60).sum()),
        "anomaly_count": int(df["Anomaly"].sum()),
        "average_performance": round(float(df["PerformanceScore"].mean()), 2),
        "average_risk_score": round(float(df["CalculatedRiskScore"].mean()), 2),
        "department_summary": {},
        "segment_summary": {},
    }

    dept = (
        df.groupby("Department")
        .agg(
            Employees=("EmployeeID", "count"),
            AvgPerformance=("PerformanceScore", "mean"),
            AvgRisk=("CalculatedRiskScore", "mean"),
            HighAttritionRisk=("AttritionProbability", lambda x: int((x >= 65).sum())),
            TrainingNeed=("TrainingNeedProbability", lambda x: int((x >= 60).sum())),
        )
        .round(2)
    )
    report["department_summary"] = dept.to_dict(orient="index")

    seg = (
        df.groupby("EmployeeSegment")
        .agg(
            Employees=("EmployeeID", "count"),
            AvgPerformance=("PerformanceScore", "mean"),
            AvgRisk=("CalculatedRiskScore", "mean"),
        )
        .round(2)
    )
    report["segment_summary"] = seg.to_dict(orient="index")

    os.makedirs(os.path.join(BASE_DIR, "outputs", "reports"), exist_ok=True)
    safe_json_dump(report, os.path.join(BASE_DIR, "outputs", "reports", "workforce_report.json"))

    with open(os.path.join(BASE_DIR, "outputs", "reports", "workforce_report.txt"), "w", encoding="utf-8") as f:
        f.write("AI-BASED EMPLOYEE & WORKFORCE INTELLIGENCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Employees: {report['total_employees']}\n")
        f.write(f"High Attrition Risk: {report['high_attrition_risk_count']}\n")
        f.write(f"Medium Attrition Risk: {report['medium_attrition_risk_count']}\n")
        f.write(f"Promotion Suitable: {report['promotion_suitable_count']}\n")
        f.write(f"Training Need: {report['training_need_count']}\n")
        f.write(f"Anomalies: {report['anomaly_count']}\n")
        f.write(f"Average Performance: {report['average_performance']}/5\n")
        f.write(f"Average Risk Score: {report['average_risk_score']}/100\n\n")

        f.write("DEPARTMENT SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(dept.to_string())
        f.write("\n\nSEGMENT SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(seg.to_string())
        f.write("\n")

    print("\nWorkforce report created:")
    print("  outputs/reports/workforce_report.txt")
    print("  outputs/reports/workforce_report.json")
    return report


if __name__ == "__main__":
    create_workforce_report()
