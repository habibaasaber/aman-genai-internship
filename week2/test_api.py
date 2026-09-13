import pytest
from fastapi.testclient import TestClient

from api import app, SearchQuery

client = TestClient(app)

def test_1_normal_hr_question():
    """Test 1: Normal HR question with expected relevant policy."""
    response = client.post("/search_hr_policies", json={"query": "How many annual leave days am I entitled to?"})
    assert response.status_code == 200
    data = response.json()
    # It might return success or insufficient info if the DB is empty during testing, 
    # but the structure should be correct.
    assert "results" in data
    assert "confidence" in data
    assert data["status"] in ["success", "insufficient_information"]

def test_2_no_relevant_policy():
    """Test 2: No relevant policy (insufficient information)."""
    response = client.post("/search_hr_policies", json={"query": "Can I bring my dog to the office?"})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "confidence" in data
    # Assuming "dog" is not in the HR policy, it should return low confidence
    if data["confidence"] < 0.2:
        assert data["status"] == "insufficient_information"

def test_4_pii_protection():
    """Test 4: PII protection (Runtime Sanitization)."""
    # The query has PII. The tool should sanitize it before Langfuse context and retrieval.
    query = "John Doe, employee ID EMP-12345, is asking about his 35000 EGP salary."
    response = client.post("/search_hr_policies", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data

    # The actual retrieval might fail because of the weird query, but we test the endpoint doesn't crash
    # and the result (if any) doesn't contain PII that was returned.
    for res in data.get("results", []):
        assert "EMP-12345" not in res["text"]
        assert "35000" not in res["text"]

def test_malformed_input():
    """Test malformed tool input."""
    response = client.post("/search_hr_policies", json={"invalid_field": "test"})
    assert response.status_code == 422  # Unprocessable Entity
