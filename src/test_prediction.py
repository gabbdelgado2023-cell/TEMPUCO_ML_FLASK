from __future__ import annotations

import json
import sys
from typing import Any

import requests
from requests import Response
from requests.exceptions import ConnectionError, RequestException, Timeout


# Change this value if your Flask API uses another port.
BASE_URL = "http://127.0.0.1:5055"

HEALTH_ENDPOINT = f"{BASE_URL}/health"
LOAN_ENDPOINT = f"{BASE_URL}/predict/loan"
SCHOLARSHIP_ENDPOINT = f"{BASE_URL}/predict/scholarship"

REQUEST_TIMEOUT = 10


LOAN_TEST_DATA: dict[str, Any] = {
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


SCHOLARSHIP_TEST_DATA: dict[str, Any] = {
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


def print_section(title: str) -> None:
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_json(data: Any) -> None:
    """Print dictionaries or lists as readable JSON."""

    print(
        json.dumps(
            data,
            indent=4,
            ensure_ascii=False,
        )
    )


def get_response_data(response: Response) -> Any:
    """
    Return the response as JSON.

    If the server did not return valid JSON, return its raw text.
    """

    try:
        return response.json()
    except ValueError:
        return {
            "error": "The API did not return valid JSON.",
            "raw_response": response.text,
        }


def check_health() -> bool:
    """Check whether the Flask API and both models are available."""

    print_section("TEMPUCO ML API HEALTH CHECK")

    try:
        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=REQUEST_TIMEOUT,
        )

        response_data = get_response_data(response)

        print(f"URL: {HEALTH_ENDPOINT}")
        print(f"HTTP status: {response.status_code}")
        print()
        print_json(response_data)

        if response.status_code != 200:
            print()
            print("Health check failed.")
            return False

        if not isinstance(response_data, dict):
            print()
            print("Unexpected health-check response.")
            return False

        api_status = response_data.get("status")
        loan_loaded = response_data.get("loan_model_loaded")
        scholarship_loaded = response_data.get(
            "scholarship_model_loaded"
        )

        if api_status != "ok":
            print()
            print("The API did not report an OK status.")
            return False

        if loan_loaded is not True:
            print()
            print("The loan model was not loaded.")
            return False

        if scholarship_loaded is not True:
            print()
            print("The scholarship model was not loaded.")
            return False

        print()
        print("Health check passed. Both models are loaded.")
        return True

    except ConnectionError:
        print(
            f"Could not connect to {BASE_URL}.\n"
            "Make sure the Flask API is running using:\n"
            "python src/api.py"
        )
        return False

    except Timeout:
        print("The health-check request timed out.")
        return False

    except RequestException as error:
        print(f"Health-check request failed: {error}")
        return False


def test_prediction(
    title: str,
    endpoint: str,
    payload: dict[str, Any],
) -> bool:
    """Send sample applicant data to a prediction endpoint."""

    print_section(title)

    print(f"URL: {endpoint}")
    print()
    print("Request data:")
    print_json(payload)

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response_data = get_response_data(response)

        print()
        print(f"HTTP status: {response.status_code}")
        print()
        print("API response:")
        print_json(response_data)

        if response.status_code != 200:
            print()
            print("Prediction request failed.")
            return False

        if not isinstance(response_data, dict):
            print()
            print("Unexpected API response.")
            return False

        prediction = response_data.get("prediction")
        confidence = response_data.get("confidence")
        probabilities = response_data.get("probabilities")

        if prediction is None:
            print()
            print("The response does not contain a prediction.")
            return False

        print()
        print(f"Prediction: {prediction}")

        if isinstance(confidence, (int, float)):
            print(f"Confidence: {confidence:.2%}")
        else:
            print(f"Confidence: {confidence}")

        if isinstance(probabilities, dict):
            print()
            print("Class probabilities:")

            sorted_probabilities = sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for class_name, probability in sorted_probabilities:
                if isinstance(probability, (int, float)):
                    print(
                        f"  {class_name}: "
                        f"{float(probability):.2%}"
                    )
                else:
                    print(f"  {class_name}: {probability}")

        print()
        print("Prediction test passed.")
        return True

    except ConnectionError:
        print()
        print(
            f"Could not connect to {BASE_URL}.\n"
            "Make sure the API is running in another terminal."
        )
        return False

    except Timeout:
        print()
        print("The prediction request timed out.")
        return False

    except RequestException as error:
        print()
        print(f"Prediction request failed: {error}")
        return False


def main() -> None:
    """Run the complete API test."""

    print_section("TEMPUCO AIDSUITE DECISION TREE API TEST")

    print(f"API address: {BASE_URL}")
    print(
        "Make sure python src/api.py is running "
        "in another PyCharm terminal."
    )

    health_passed = check_health()

    if not health_passed:
        print_section("TEST STOPPED")
        print(
            "The prediction tests were not executed because "
            "the API health check failed."
        )
        sys.exit(1)

    loan_test_passed = test_prediction(
        title="LOAN RISK PREDICTION TEST",
        endpoint=LOAN_ENDPOINT,
        payload=LOAN_TEST_DATA,
    )

    scholarship_test_passed = test_prediction(
        title="SCHOLARSHIP ELIGIBILITY PREDICTION TEST",
        endpoint=SCHOLARSHIP_ENDPOINT,
        payload=SCHOLARSHIP_TEST_DATA,
    )

    print_section("FINAL TEST RESULT")

    print(
        "Loan prediction: "
        f"{'PASSED' if loan_test_passed else 'FAILED'}"
    )

    print(
        "Scholarship prediction: "
        f"{'PASSED' if scholarship_test_passed else 'FAILED'}"
    )

    if loan_test_passed and scholarship_test_passed:
        print()
        print("All TEMPUCO machine-learning API tests passed.")
        print(
            "The local Python API is ready for Laravel integration."
        )
        sys.exit(0)

    print()
    print("One or more API tests failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()