import pytest
import numpy as np
import pandas as pd
import sys
from model.train_model import (
    train_model, compute_model_metrics, inference, process_data, CAT_FEATURES
)
sys.path.insert(0, ".")


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "workclass": np.random.choice(["Private", "Self-emp-not-inc"], n),
        "fnlgt": np.random.randint(100000, 500000, n),
        "education": np.random.choice(["Bachelors", "HS-grad", "Some-college"], n),
        "education-num": np.random.randint(8, 16, n),
        "marital-status": np.random.choice(["Married-civ-spouse", "Never-married"], n),
        "occupation": np.random.choice(["Exec-managerial", "Craft-repair"], n),
        "relationship": np.random.choice(["Husband", "Not-in-family"], n),
        "race": np.random.choice(["White", "Black"], n),
        "sex": np.random.choice(["Male", "Female"], n),
        "capital-gain": np.zeros(n, dtype=int),
        "capital-loss": np.zeros(n, dtype=int),
        "hours-per-week": np.random.randint(20, 60, n),
        "native-country": np.random.choice(["United-States", "Mexico"], n),
        "salary": np.random.choice(["<=50K", ">50K"], n),
    })
    return df


def test_train_model_returns_model(sample_data):
    X, y, enc, _ = process_data(
        sample_data, CAT_FEATURES, "salary", training=True
    )
    model = train_model(X, y)
    assert hasattr(model, "predict")


def test_inference_output_shape(sample_data):
    X, y, enc, _ = process_data(
        sample_data, CAT_FEATURES, "salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape


def test_compute_model_metrics_range(sample_data):
    X, y, enc, _ = process_data(
        sample_data, CAT_FEATURES, "salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    p, r, f = compute_model_metrics(y, preds)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f <= 1.0
