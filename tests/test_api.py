from fastapi.testclient import TestClient
import sys
from main import app
sys.path.insert(0, ".")


client = TestClient(app)


def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()
    assert r.json()["message"] == "Welcome to the Census Salary Prediction API!"


def test_post_predict_above_50k():
    payload = {
        "age": 52, "workclass": "Self-emp-not-inc", "fnlgt": 209642,
        "education": "HS-grad", "education-num": 9,
        "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male",
        "capital-gain": 0, "capital-loss": 0,
        "hours-per-week": 45, "native-country": "United-States"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] in ["<=50K", ">50K"]


def test_post_predict_below_50k():
    payload = {
        "age": 25, "workclass": "Private", "fnlgt": 226956,
        "education": "11th", "education-num": 7,
        "marital-status": "Never-married", "occupation": "Machine-op-inspct",
        "relationship": "Own-child", "race": "Black", "sex": "Male",
        "capital-gain": 0, "capital-loss": 0,
        "hours-per-week": 40, "native-country": "United-States"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"
