import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
from src.generate_dataset import generate_employee_dataset
from src.train_models import run_training
from src.workforce_report import create_workforce_report


def main():
    print("=" * 70)
    print("AI-BASED EMPLOYEE & WORKFORCE INTELLIGENCE")
    print("Complete Machine Learning Pipeline")
    print("=" * 70)

    data_path = "data/employee_data.csv"

    if not os.path.exists(data_path):
        generate_employee_dataset(n=600, path=data_path, random_state=42)
    else:
        print(f"Using existing dataset: {data_path}")

    df, metrics = run_training()
    report = create_workforce_report()

    first_employee = df.iloc[0]["EmployeeID"]

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Employees processed: {len(df)}")
    print(f"Example employee: {first_employee}")
    print("\nMetrics:")
    for name, values in metrics.items():
        print(f"  {name}: {values}")

    print("\nGenerated outputs:")
    print("  - Model files: outputs/models/")
    print("  - Predictions: outputs/predictions/employee_predictions.csv")
    print("  - Charts: outputs/charts/")
    print("  - Workforce report: outputs/reports/")
    print("\nTo inspect one employee:")
    print(f"  python src/predict_employee.py --employee-id {first_employee}")
    print("\nNOTE: This project uses synthetic demo data by default.")


if __name__ == "__main__":
    main()
