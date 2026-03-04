import pytest
import sys
import pandas as pd
from model.train_model import (
    train_model, compute_model_metrics, inference, process_data, CAT_FEATURES
)
sys.path.insert(0, ".")


@pytest.fixture
def sample_data():
    df = pd.read_csv("data/census_clean.csv")
    return df.head(200)


def test_train_model_returns_model(sample_data):
    X, y, enc, _ = process_data(sample_data, CAT_FEATURES, "salary", training=True)
    model = train_model(X, y)
    assert hasattr(model, "predict")


def test_inference_output_shape(sample_data):
    X, y, enc, _ = process_data(sample_data, CAT_FEATURES, "salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape


def test_compute_model_metrics_range(sample_data):
    X, y, enc, _ = process_data(sample_data, CAT_FEATURES, "salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    p, r, f = compute_model_metrics(y, preds)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f <= 1.0
