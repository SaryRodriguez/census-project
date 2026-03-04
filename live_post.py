import requests

url = "https://census-project-7yl0.onrender.com/predict"

payload = {
    "age": 37, "workclass": "Private", "fnlgt": 280464,
    "education": "Some-college", "education-num": 10,
    "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
    "relationship": "Husband", "race": "Black", "sex": "Male",
    "capital-gain": 0, "capital-loss": 0,
    "hours-per-week": 80, "native-country": "United-States"
}

r = requests.post(url, json=payload)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")
