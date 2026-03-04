import pandas as pd

cols = [
    "age", "workclass", "fnlgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"
]

df = pd.read_csv("data/census.csv", header=None, names=cols)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.to_csv("data/census_clean.csv", index=False)
print(df.head())
