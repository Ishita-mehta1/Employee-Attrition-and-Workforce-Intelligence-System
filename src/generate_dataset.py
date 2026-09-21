import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
DEFAULT_PATH = os.path.join("data", "employee_data.csv")


def generate_employee_dataset(n=600, path=DEFAULT_PATH, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)

    departments = np.array(["Sales", "IT", "HR", "Finance", "Marketing", "Operations"])
    roles = np.array([
        "Executive", "Analyst", "Developer", "Manager",
        "Specialist", "Coordinator"
    ])
    education = np.array(["Bachelor", "Master", "PhD"])
    cities = np.array(["Delhi", "Dehradun", "Noida", "Haldwani", "Lucknow", "Jaipur"])

    department = rng.choice(departments, n)
    job_role = rng.choice(roles, n)
    education_level = rng.choice(education, n, p=[0.52, 0.40, 0.08])
    city = rng.choice(cities, n)

    age = rng.integers(21, 61, n)
    monthly_income = np.clip(rng.normal(52000, 22000, n), 18000, 150000).round(0)
    years_at_company = np.clip(rng.normal(5.2, 4.2, n), 0.2, 25).round(1)
    job_satisfaction = rng.integers(1, 6, n)
    work_life_balance = rng.integers(1, 6, n)
    overtime = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    performance_score = np.clip(
        2.8
        + 0.35 * job_satisfaction
        + 0.25 * work_life_balance
        + 0.08 * years_at_company
        + rng.normal(0, 0.7, n),
        1, 5
    ).round(2)
    training_hours = np.clip(rng.normal(30, 14, n), 5, 90).round(1)
    projects_completed = np.clip(
        rng.poisson(5, n) + (performance_score > 4).astype(int),
        0, 15
    )

    # Targets are generated from transparent synthetic rules so the demo
    # has realistic relationships while remaining fully reproducible.
    overtime_num = (overtime == "Yes").astype(int)
    attrition_probability = (
        -0.7
        + 0.55 * overtime_num
        - 0.35 * job_satisfaction
        - 0.20 * work_life_balance
        - 0.06 * years_at_company
        - 0.000006 * monthly_income
        + 0.10 * (age < 27)
        + rng.normal(0, 0.7, n)
    )
    attrition = (attrition_probability > -1.8).astype(int)

    performance_target = np.clip(
        performance_score
        + 0.015 * projects_completed
        + 0.004 * training_hours
        + rng.normal(0, 0.25, n),
        1, 5
    )

    promotion_score = (
        0.30 * (performance_score / 5)
        + 0.20 * (years_at_company / 10)
        + 0.15 * (training_hours / 60)
        + 0.15 * (projects_completed / 10)
        + 0.10 * (job_satisfaction / 5)
        + 0.10 * (work_life_balance / 5)
    )
    promotion = (promotion_score >= 0.56).astype(int)

    training_need_score = (
        0.40 * (performance_score < 3).astype(int)
        + 0.25 * (training_hours < 20).astype(int)
        + 0.20 * (job_satisfaction <= 2).astype(int)
        + 0.15 * (projects_completed < 3).astype(int)
    )
    training_need = (training_need_score >= 0.40).astype(int)

    risk_raw = (
        0.35 * attrition
        + 0.20 * (performance_score < 3).astype(int)
        + 0.15 * (job_satisfaction <= 2).astype(int)
        + 0.15 * overtime_num
        + 0.15 * (work_life_balance <= 2).astype(int)
    )
    employee_risk_score = np.clip(
        100 * risk_raw + rng.normal(0, 5, n), 0, 100
    ).round(2)

    df = pd.DataFrame({
        "EmployeeID": [f"EMP{1001+i}" for i in range(n)],
        "Name": [f"Employee_{1001+i}" for i in range(n)],
        "Age": age,
        "Department": department,
        "JobRole": job_role,
        "EducationLevel": education_level,
        "City": city,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "Overtime": overtime,
        "PerformanceScore": performance_score,
        "TrainingHours": training_hours,
        "ProjectsCompleted": projects_completed,
        "Attrition": attrition,
        "PerformanceTarget": performance_target.round(2),
        "PromotionSuitable": promotion,
        "TrainingNeed": training_need,
        "EmployeeRiskScore": employee_risk_score,
    })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Created {len(df)} synthetic employee records at {path}")
    return df


if __name__ == "__main__":
    generate_employee_dataset()
