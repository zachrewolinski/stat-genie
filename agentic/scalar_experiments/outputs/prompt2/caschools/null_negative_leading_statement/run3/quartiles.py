import pandas as pd


df = pd.read_csv("caschools.csv")
df["str"] = df["students"] / df["teachers"]
df["testscr"] = (df["read"] + df["math"]) / 2.0

# Define quartiles of student-teacher ratio
quartiles = pd.qcut(df["str"], 4, labels=["Q1_lowest_STR", "Q2", "Q3", "Q4_highest_STR"])

summary = df.groupby(quartiles)["testscr"].agg(["mean", "std", "count"])
print(summary)
