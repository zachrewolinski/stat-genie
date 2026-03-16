import pandas as pd
import numpy as np


df = pd.read_csv("panda_nuts.csv")

df["efficiency"] = df["feature5"] / df["feature6"]

# Efficiency summary overall
print("Overall efficiency mean:", df["efficiency"].mean())
print("Overall efficiency median:", df["efficiency"].median())

# By sex
print("Efficiency by sex (mean, median, n):")
for sex, sub in df.groupby("feature3"):
    print(sex, sub["efficiency"].mean(), sub["efficiency"].median(), len(sub))

# By help
print("Efficiency by help (mean, median, n):")
for helpv, sub in df.groupby("feature7"):
    print(helpv, sub["efficiency"].mean(), sub["efficiency"].median(), len(sub))

# Age correlation
print("Age-efficiency Pearson:", df[["feature2","efficiency"]].corr().iloc[0,1])
