import pandas as pd


df = pd.read_csv("caschools.csv")
df["str"] = df["students"] / df["teachers"]
df["testscr"] = (df["read"] + df["math"]) / 2.0

print(df[["str", "testscr"]].describe())
print("\nCorrelation matrix:\n", df[["str", "testscr", "income", "calworks", "lunch", "english"]].corr())
