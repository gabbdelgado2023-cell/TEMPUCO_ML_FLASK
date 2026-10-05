from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request

ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = ROOT_DIRECTORY / "models"

LOAN_MODEL_PATH = MODEL_DIRECTORY / "loan_risk_model.joblib"
SCHOLARSHIP_MODEL_PATH = MODEL_DIRECTORY / "scholarship_model.joblib"

LOAN_NUMERIC_FIELDS = {
    "age": int,
    "monthly_income": float,
    "years_employed": float,
    "existing_loan_balance": float,
    "monthly_debt_payment": float,
    "requested_amount": float,
    "term_months": int,
    "late_payments_12m": int,
    "savings_balance": float,
}

LOAN_CATEGORY_FIELDS = {
    "employment_status": {
        "regular",
        "contractual",
        "self_employed",
        "unemployed",
    },
    "has_co_maker": {"yes", "no"},
}

SCHOLARSHIP_NUMERIC_FIELDS = {
    "general_average": float,
    "family_monthly_income": float,
    "household_size": int,
    "attendance_rate": float,
    "disciplinary_incidents": int,
    "community_service_hours": float,
}

SCHOLARSHIP_CATEGORY_FIELDS = {
    "current_year_level": {
        "grade_7",
        "grade_8",
        "grade_9",
        "grade_10",
        "grade_11",
        "grade_12",
    },
    "complete_documents": {"yes", "no"},
    "previous_scholarship": {"yes", "no"},
}


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run 'python src/train_models.py' first."
        )

    return joblib.load(model_path)


def normalize_category(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"

    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def validate_and_prepare_payload(
    payload: Any,
    numeric_fields: dict[str, type],
    category_fields: dict[str, set[str]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(payload, dict):
        return None, {
            "error": "The request body must contain a valid JSON object."
        }

    required_fields = list(numeric_fields) + list(category_fields)
    missing_fields = [
        field
        for field in required_fields
        if field not in payload or payload[field] in (None, "")
    ]

    if missing_fields:
        return None, {
            "error": "Missing required fields.",
            "missing_fields": missing_fields,
        }

    prepared: dict[str, Any] = {}
    invalid_fields: dict[str, str] = {}

    for field, expected_type in numeric_fields.items():
        try:
            prepared[field] = expected_type(payload[field])
        except (TypeError, ValueError):
            invalid_fields[field] = (
                f"Expected a valid {expected_type.__name__} value."
            )

    for field, allowed_values in category_fields.items():
        normalized_value = normalize_category(payload[field])

        if normalized_value not in allowed_values:
            invalid_fields[field] = (
                "Allowed values: " + ", ".join(sorted(allowed_values))
            )
        else:
            prepared[field] = normalized_value

    if invalid_fields:
        return None, {
            "error": "One or more input values are invalid.",
            "invalid_fields": invalid_fields,
        }

    return prepared, None


def make_prediction(model, record: dict[str, Any]) -> dict[str, Any]:
    input_frame = pd.DataFrame([record])

    prediction = str(model.predict(input_frame)[0])
    probabilities = model.predict_proba(input_frame)[0]
    classes = [str(class_name) for class_name in model.classes_]

    probability_map = {
        class_name: round(float(probability), 4)
        for class_name, probability in zip(classes, probabilities)
    }

    return {
        "prediction": prediction,
        "confidence": round(max(probability_map.values()), 4),
        "probabilities": probability_map,
        "advisory": (
            "The Decision Tree result is advisory only. Final approval "
            "must remain with authorized TEMPUCO personnel."
        ),
        "sample_model_warning": (
            "This model currently uses synthetic demonstration data. "
            "Retrain and validate it using authorized historical TEMPUCO "
            "records before actual deployment."
        ),
    }


loan_model = load_model(LOAN_MODEL_PATH)
scholarship_model = load_model(SCHOLARSHIP_MODEL_PATH)

app = Flask(__name__)


@app.get("/")
def index():
    return jsonify(
        {
            "service": "TEMPUCO AidSuite Decision Tree API",
            "status": "running",
            "endpoints": {
                "health": "GET /health",
                "loan_prediction": "POST /predict/loan",
                "scholarship_prediction": "POST /predict/scholarship",
            },
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "loan_model_loaded": loan_model is not None,
            "scholarship_model_loaded": scholarship_model is not None,
        }
    )


@app.post("/predict/loan")
def predict_loan():
    record, validation_error = validate_and_prepare_payload(
        payload=request.get_json(silent=True),
        numeric_fields=LOAN_NUMERIC_FIELDS,
        category_fields=LOAN_CATEGORY_FIELDS,
    )

    if validation_error:
        return jsonify(validation_error), 422

    try:
        return jsonify(make_prediction(loan_model, record))
    except (TypeError, ValueError) as error:
        return jsonify(
            {
                "error": "The loan model could not process the input.",
                "details": str(error),
            }
        ), 422


@app.post("/predict/scholarship")
def predict_scholarship():
    record, validation_error = validate_and_prepare_payload(
        payload=request.get_json(silent=True),
        numeric_fields=SCHOLARSHIP_NUMERIC_FIELDS,
        category_fields=SCHOLARSHIP_CATEGORY_FIELDS,
    )

    if validation_error:
        return jsonify(validation_error), 422

    try:
        return jsonify(make_prediction(scholarship_model, record))
    except (TypeError, ValueError) as error:
        return jsonify(
            {
                "error": (
                    "The scholarship model could not process the input."
                ),
                "details": str(error),
            }
        ), 422


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5055,
        debug=True,
        use_reloader=False,
    )