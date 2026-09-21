import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, mean_absolute_error,
    mean_squared_error, r2_score, silhouette_score
)

from src.utils import (
    MODEL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    ensure_dirs, feature_importance_table
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "employee_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
PRED_DIR = os.path.join(BASE_DIR, "outputs", "predictions")


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )


def classifier_pipeline():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ))
    ])


def regressor_pipeline():
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=250, random_state=42
        ))
    ])


def train_classifier(df, target, filename):
    X = df[MODEL_FEATURES]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipe = classifier_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print(f"{target}: accuracy={acc:.3f}")
    print(classification_report(y_test, pred, zero_division=0))

    joblib.dump(pipe, os.path.join(MODEL_DIR, filename))
    return pipe, {"accuracy": round(float(acc), 4)}


def train_regression(df, target, filename):
    X = df[MODEL_FEATURES]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipe = regressor_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"{target}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

    joblib.dump(pipe, os.path.join(MODEL_DIR, filename))
    return pipe, {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }


def train_clustering(df):
    X = df[NUMERIC_FEATURES].copy()
    X_scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, clusters)

    df["EmployeeSegment"] = clusters
    joblib.dump(model, os.path.join(MODEL_DIR, "kmeans_segmentation.joblib"))

    print(f"K-Means silhouette score: {score:.3f}")
    return df, {"silhouette_score": round(float(score), 4)}


def train_anomaly_model(df):
    X = df[NUMERIC_FEATURES].copy()
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        contamination=0.05, random_state=42, n_estimators=200
    )
    anomaly = model.fit_predict(X_scaled)
    df["Anomaly"] = np.where(anomaly == -1, 1, 0)

    joblib.dump(model, os.path.join(MODEL_DIR, "anomaly_detector.joblib"))
    return df


def save_global_importance(pipe, filename):
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocessor"]
    table = feature_importance_table(model, preprocessor)
    table.to_csv(os.path.join(PRED_DIR, filename), index=False)


def create_charts(df):
    os.makedirs(os.path.join(BASE_DIR, "outputs", "charts"), exist_ok=True)

    plt.figure(figsize=(8, 5))
    df["Attrition"].value_counts().sort_index().plot(kind="bar")
    plt.xticks([0, 1], ["No", "Yes"], rotation=0)
    plt.title("Employee Attrition Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "outputs", "charts", "attrition_distribution.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    df["EmployeeRiskScore"].plot(kind="hist", bins=20)
    plt.title("Employee Risk Score Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "outputs", "charts", "risk_distribution.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for segment in sorted(df["EmployeeSegment"].unique()):
        subset = df[df["EmployeeSegment"] == segment]
        plt.scatter(subset["YearsAtCompany"], subset["PerformanceScore"], label=f"Segment {segment}")
    plt.legend()
    plt.title("Employee Segments")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "outputs", "charts", "employee_segments.png"), dpi=160)
    plt.close()


def run_training():
    ensure_dirs()
    df = pd.read_csv(DATA_PATH)

    metrics = {}

    attrition_model, metrics["attrition"] = train_classifier(
        df, "Attrition", "attrition_model.joblib"
    )
    promotion_model, metrics["promotion"] = train_classifier(
        df, "PromotionSuitable", "promotion_model.joblib"
    )
    training_model, metrics["training_need"] = train_classifier(
        df, "TrainingNeed", "training_need_model.joblib"
    )
    performance_model, metrics["performance"] = train_regression(
        df, "PerformanceTarget", "performance_model.joblib"
    )

    df["PredictedAttrition"] = attrition_model.predict(df[MODEL_FEATURES])
    df["AttritionProbability"] = (
        attrition_model.predict_proba(df[MODEL_FEATURES])[:, 1] * 100
    ).round(2)

    df["PredictedPromotion"] = promotion_model.predict(df[MODEL_FEATURES])
    df["PromotionProbability"] = (
        promotion_model.predict_proba(df[MODEL_FEATURES])[:, 1] * 100
    ).round(2)

    df["PredictedTrainingNeed"] = training_model.predict(df[MODEL_FEATURES])
    df["TrainingNeedProbability"] = (
        training_model.predict_proba(df[MODEL_FEATURES])[:, 1] * 100
    ).round(2)

    df["PredictedPerformance"] = (
        performance_model.predict(df[MODEL_FEATURES]).round(2)
    )

    df, metrics["segmentation"] = train_clustering(df)
    df = train_anomaly_model(df)

    # Combine model signals into a transparent workforce risk score.
    df["CalculatedRiskScore"] = (
        0.45 * df["AttritionProbability"]
        + 0.20 * (100 - df["PerformanceScore"] * 20)
        + 0.15 * (100 - df["JobSatisfaction"] * 20)
        + 0.10 * (100 - df["WorkLifeBalance"] * 20)
        + 0.10 * df["TrainingNeedProbability"]
    ).clip(0, 100).round(2)

    df.to_csv(
        os.path.join(PRED_DIR, "employee_predictions.csv"),
        index=False
    )

    save_global_importance(
        attrition_model, "attrition_global_feature_importance.csv"
    )
    save_global_importance(
        promotion_model, "promotion_global_feature_importance.csv"
    )
    save_global_importance(
        training_model, "training_need_global_feature_importance.csv"
    )

    create_charts(df)

    joblib.dump(
        {"features": MODEL_FEATURES, "metrics": metrics},
        os.path.join(MODEL_DIR, "project_metadata.joblib")
    )

    return df, metrics


if __name__ == "__main__":
    run_training()
