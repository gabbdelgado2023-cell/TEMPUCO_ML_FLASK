from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = ROOT_DIRECTORY / "models"

LOAN_MODEL_PATH = MODEL_DIRECTORY / "loan_risk_model.joblib"
SCHOLARSHIP_MODEL_PATH = MODEL_DIRECTORY / "scholarship_model.joblib"


LOAN_SAMPLE: dict[str, Any] = {
    "age": 38,
    "employment_status": "regular",
    "monthly_income": 42000,
    "years_employed": 8,
    "existing_loan_balance": 15000,
    "monthly_debt_payment": 5500,
    "requested_amount": 50000,
    "term_months": 12,
    "late_payments_12m": 0,
    "savings_balance": 65000,
    "has_co_maker": "yes",
}


SCHOLARSHIP_SAMPLE: dict[str, Any] = {
    "general_average": 91.5,
    "family_monthly_income": 18000,
    "household_size": 6,
    "attendance_rate": 97,
    "disciplinary_incidents": 0,
    "community_service_hours": 30,
    "current_year_level": "grade_11",
    "complete_documents": "yes",
    "previous_scholarship": "no",
}


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run 'python src/train_models.py' first."
        )

    return joblib.load(model_path)


def display_prediction(
    model_name: str,
    model,
    sample: dict[str, Any],
) -> None:
    input_frame = pd.DataFrame([sample])

    prediction = str(model.predict(input_frame)[0])
    probabilities = model.predict_proba(input_frame)[0]

    classes = [
        str(class_name)
        for class_name in model.classes_
    ]

    print("=" * 65)
    print(f"{model_name.upper()} PREDICTION")
    print("=" * 65)

    print(f"Result: {prediction}")
    print("Probabilities:")

    for class_name, probability in zip(classes, probabilities):
        print(f"  {class_name}: {float(probability):.2%}")

    print()


def main() -> None:
    loan_model = load_model(LOAN_MODEL_PATH)
    scholarship_model = load_model(SCHOLARSHIP_MODEL_PATH)

    display_prediction(
        model_name="Loan risk",
        model=loan_model,
        sample=LOAN_SAMPLE,
    )

    display_prediction(
        model_name="Scholarship eligibility",
        model=scholarship_model,
        sample=SCHOLARSHIP_SAMPLE,
    )

    print(
        "Reminder: The predictions are advisory and currently "
        "use synthetic demonstration data."
    )


if __name__ == "__main__":
    main()