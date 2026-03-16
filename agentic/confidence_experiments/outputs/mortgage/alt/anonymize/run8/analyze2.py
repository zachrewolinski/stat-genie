import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

df = pd.read_csv("mortgage.csv")

ct = pd.crosstab(df["feature2"], df["feature14"], dropna=False)
print(ct)

# logistic with gender only
X = sm.add_constant(df[["feature2"]])
y = df["feature14"]
logit = sm.Logit(y, X).fit(disp=False)
print(logit.summary2().tables[1])

# check if any missing
print("missing feature2", df["feature2"].isna().sum())
print("missing feature14", df["feature14"].isna().sum())

# check complement
print("feature11+feature14 unique", (df["feature11"] + df["feature14"]).unique())
