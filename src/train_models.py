from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Ensures repeatable training and testing results.
RANDOM_STATE = 42

# Main project directory.
ROOT_DIRECTORY = Path(__file__).resolve().parents[1]

DATA_DIRECTORY = ROOT_DIRECTORY / "data"
MODEL_DIRECTORY = ROOT_DIRECTORY / "models"
OUTPUT_DIRECTORY = ROOT_DIRECTORY / "outputs"


MODEL_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "loan": {
        "csv_file": DATA_DIRECTORY / "loan_training_data_tempuco.csv",
        "target_column": "loan_risk",

        "categorical_features": [
            "employment_status",
            "has_co_maker",
        ],

        "numeric_features": [
            "age",
            "monthly_income",
            "years_employed",
            "existing_loan_balance",
            "monthly_debt_payment",
            "requested_amount",
            "term_months",
            "late_payments_12m",
            "savings_balance",
        ],

        "model_file": MODEL_DIRECTORY / "loan_risk_model.joblib",
        "metrics_file": OUTPUT_DIRECTORY / "loan_metrics.json",
        "tree_image": OUTPUT_DIRECTORY / "loan_decision_tree.png",

        "max_depth": 6,
        "min_samples_leaf": 12,
    },

    "scholarship": {
        "csv_file": DATA_DIRECTORY / "scholarship_training_data_tempuco.csv",
        "target_column": "scholarship_result",

        "categorical_features": [
            "current_year_level",
            "complete_documents",
            "previous_scholarship",
        ],

        "numeric_features": [
            "general_average",
            "family_monthly_income",
            "household_size",
            "attendance_rate",
            "disciplinary_incidents",
            "community_service_hours",
        ],

        "model_file": MODEL_DIRECTORY / "scholarship_model.joblib",
        "metrics_file": OUTPUT_DIRECTORY / "scholarship_metrics.json",
        "tree_image": OUTPUT_DIRECTORY / "scholarship_decision_tree.png",

        "max_depth": 6,
        "min_samples_leaf": 12,
    },
}


def create_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
    max_depth: int,
    min_samples_leaf: int,
) -> Pipeline:
    """
    Creates the preprocessing and Decision Tree pipeline.

    Text values are converted using OneHotEncoder.
    Numeric values are passed directly to the model.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    decision_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", decision_tree),
        ]
    )

    return pipeline


def train_model(
    model_name: str,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """
    Loads the CSV file, divides the dataset, trains the model,
    evaluates it, and saves the trained model.
    """

    csv_file = configuration["csv_file"]
    target_column = configuration["target_column"]

    categorical_features = configuration["categorical_features"]
    numeric_features = configuration["numeric_features"]

    all_features = categorical_features + numeric_features

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Training file was not found: {csv_file}"
        )

    print()
    print("=" * 65)
    print(f"TRAINING {model_name.upper()} MODEL")
    print("=" * 65)

    # Read the CSV dataset.
    dataset = pd.read_csv(csv_file)

    print(f"Dataset: {csv_file.name}")
    print(f"Total records: {len(dataset)}")

    # Check whether all expected columns exist.
    required_columns = all_features + [target_column]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The {model_name} dataset is missing these columns: "
            f"{missing_columns}"
        )

    # Remove records that have missing target results.
    dataset = dataset.dropna(subset=[target_column])

    # Input fields.
    X = dataset[all_features]

    # Correct answers or labels.
    y = dataset[target_column]

    print()
    print("Target distribution:")
    print(y.value_counts())

    # 80% training data and 20% testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print()
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    pipeline = create_pipeline(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        max_depth=configuration["max_depth"],
        min_samples_leaf=configuration["min_samples_leaf"],
    )

    # Train the preprocessing and Decision Tree.
    pipeline.fit(X_train, y_train)

    # Predict the testing records.
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report_dictionary = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    class_names = list(pipeline.classes_)

    confusion_matrix_result = confusion_matrix(
        y_test,
        predictions,
        labels=class_names,
    )

    print()
    print(f"{model_name.title()} model accuracy: {accuracy:.2%}")

    print()
    print("Classification report:")
    print(report_text)

    print("Confusion matrix:")
    print(confusion_matrix_result)

    # Create output folders if they do not exist.
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Save the complete preprocessing and model pipeline.
    joblib.dump(
        pipeline,
        configuration["model_file"],
    )

    print()
    print(f"Saved model: {configuration['model_file']}")

    metrics = {
        "model_name": model_name,
        "sample_data_only": True,
        "total_records": int(len(dataset)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "accuracy": round(float(accuracy), 4),
        "classes": class_names,
        "confusion_matrix": confusion_matrix_result.tolist(),
        "classification_report": report_dictionary,
        "input_features": all_features,
    }

    # Save the evaluation results as JSON.
    with open(
        configuration["metrics_file"],
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    print(f"Saved metrics: {configuration['metrics_file']}")

    # Get the transformed column names after encoding categories.
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    classifier = pipeline.named_steps["classifier"]

    # Generate a visual image of the Decision Tree.
    plt.figure(figsize=(26, 14))

    plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=[
            str(class_name)
            for class_name in classifier.classes_
        ],
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=4,
    )

    plt.title(
        f"TEMPUCO {model_name.title()} Decision Tree"
    )

    plt.tight_layout()

    plt.savefig(
        configuration["tree_image"],
        dpi=160,
    )

    plt.close()

    print(f"Saved tree image: {configuration['tree_image']}")

    return metrics


def main() -> None:
    """
    Trains the loan model and scholarship model.
    """

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    training_results = {}

    for model_name, configuration in MODEL_CONFIGURATIONS.items():
        result = train_model(
            model_name=model_name,
            configuration=configuration,
        )

        training_results[model_name] = {
            "accuracy": result["accuracy"],
            "classes": result["classes"],
            "total_records": result["total_records"],
        }

    summary = {
        "warning": (
            "These models use synthetic demonstration data. "
            "They must not be used for actual approval decisions "
            "until they are retrained and validated using authorized "
            "historical TEMPUCO records."
        ),
        "models": training_results,
    }

    summary_file = OUTPUT_DIRECTORY / "training_summary.json"

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print()
    print("=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)
    print(f"Training summary: {summary_file}")


if __name__ == "__main__":
    main()