# AI-Based Employee & Workforce Intelligence Using Machine Learning

A complete, runnable Python + Machine Learning project for employee/workforce intelligence.

## Core Features

1. Employee Attrition Prediction
2. Employee Performance Prediction
3. Promotion Suitability Prediction
4. Training Need Prediction
5. Employee Risk Score
6. Employee Segmentation using K-Means
7. Individual Employee Intelligence
8. Explainable AI (feature-impact explanations)
9. Anomaly Detection
10. Final Workforce Intelligence Report

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib / Seaborn (EDA/report charts)

No frontend, backend, Flask, Django, or web application is required in this version.

## Project Structure

```text
employee_workforce_intelligence/
├── data/
│   └── employee_data.csv
├── outputs/
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py
│   ├── train_models.py
│   ├── predict_employee.py
│   ├── workforce_report.py
│   └── utils.py
├── run.py
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the complete pipeline

```bash
python run.py
```

The first run creates a reproducible demo dataset of 600 employees, trains all models, generates predictions, charts, an individual employee analysis, and a workforce report.

Outputs are written to:

```text
outputs/
├── models/
├── charts/
├── predictions/
└── reports/
```

## Individual Employee Analysis

After training:

```bash
python src/predict_employee.py --employee-id EMP1001
```

You can also search by name:

```bash
python src/predict_employee.py --name Employee_101
```

## Using Your Own Dataset

Replace:

```text
data/employee_data.csv
```

with a CSV containing the required employee fields. The included generator shows the exact schema.

For a real HR dataset, do not commit personally identifiable or confidential employee information to a public GitHub repository.

## Important Note

The included dataset is synthetic/demo data generated for software demonstration. It is not real employee data and should not be used for actual HR decisions.

For real use, labels such as attrition, promotion suitability, and training need should come from properly defined organizational criteria and validated with HR/domain experts.


## GitHub Upload

Upload the project folder contents to a GitHub repository. Do not upload real/confidential HR records.

Recommended first-run commands:

```bash
python -m pip install -r requirements.txt
python run.py
python src/predict_employee.py --employee-id EMP1001
```

The project creates its model files and reports locally after `python run.py`.
