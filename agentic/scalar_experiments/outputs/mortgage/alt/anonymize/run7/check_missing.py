import pandas as pd

df = pd.read_csv("mortgage.csv")
print(df.shape)
print(df[["feature2","feature14"]].isna().sum())
print(df["feature2"].value_counts(dropna=False))
print(df["feature14"].value_counts(dropna=False))
