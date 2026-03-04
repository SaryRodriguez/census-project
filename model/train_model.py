import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score, precision_score, recall_score
import pickle
import os

CAT_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

def load_data(path):
    df = pd.read_csv(path)
    return df

def process_data(X, categorical_features, label, training=True, encoder=None, lb=None):
    y = X[label].map({"<=50K": 0, ">50K": 1}).values
    X = X.drop(label, axis=1)

    if training:
        encoder = {}
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoder[col] = le
    else:
        for col in categorical_features:
            X[col] = encoder[col].transform(X[col])

    return X.values, y, encoder, None

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    return model.predict(X)

def performance_on_slices(model, df, feature, categorical_features, encoder, label="salary"):
    results = []
    for value in df[feature].unique():
        slice_df = df[df[feature] == value].copy()
        X, y, _, _ = process_data(
            slice_df, categorical_features, label,
            training=False, encoder=encoder
        )
        preds = inference(model, X)
        p, r, f = compute_model_metrics(y, preds)
        results.append({"feature": feature, "value": value,
                        "precision": p, "recall": r, "fbeta": f})
    return results

if __name__ == "__main__":
    df = load_data("data/census_clean.csv")
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train, CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, CAT_FEATURES, label="salary", training=False, encoder=encoder
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    p, r, f = compute_model_metrics(y_test, preds)
    print(f"Precision: {p:.2f} | Recall: {r:.2f} | F1: {f:.2f}")

    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    # Slice performance
    with open("slice_output.txt", "w") as out:
        for feat in CAT_FEATURES:
            results = performance_on_slices(model, test, feat, CAT_FEATURES, encoder)
            for r in results:
                out.write(str(r) + "\n")
    print("Modelo guardado y slice_output.txt generado.")