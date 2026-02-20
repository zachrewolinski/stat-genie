import pandas as pd


df = pd.read_csv("caschools.csv")
df["str"] = df["students"] / df["teachers"]

print(df[["read", "math", "str"]].corr())
