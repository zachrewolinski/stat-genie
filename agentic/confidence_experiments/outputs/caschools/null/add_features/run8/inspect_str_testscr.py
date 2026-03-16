import pandas as pd

df = pd.read_csv("caschools.csv")
df["stratio"] = df["students"] / df["teachers"]
df["testscr"] = (df["read"] + df["math"]) / 2.0

print(df[["stratio", "testscr"]].describe())
print("\nCorrelation:", df["stratio"].corr(df["testscr"]))
