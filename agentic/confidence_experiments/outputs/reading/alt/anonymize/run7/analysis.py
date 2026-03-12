import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Basic info
print(df.head())
print(df.dtypes)

# check possible reading speed columns
# compute derived reading speed (wpm) using feature5 (reading time minus scrolling)
# and feature4 (total time)

df["wpm_feat5"] = df["feature7"] / (df["feature5"] / 60000.0)
df["wpm_feat4"] = df["feature7"] / (df["feature4"] / 60000.0)

# Compare derived wpm with feature20
for col in ["feature20", "wpm_feat5", "wpm_feat4"]:
    print(col, df[col].describe())

# correlations
print("corr feature20 vs wpm_feat5", df[["feature20","wpm_feat5"]].corr().iloc[0,1])
print("corr feature20 vs wpm_feat4", df[["feature20","wpm_feat4"]].corr().iloc[0,1])

# Dyslexia subset
sub = df[df["feature17"] == 1]

# group by reader view
for col in ["feature20", "wpm_feat5", "wpm_feat4"]:
    g0 = sub[sub["feature3"] == 0][col].dropna()
    g1 = sub[sub["feature3"] == 1][col].dropna()
    print("\n", col, "n0", len(g0), "n1", len(g1))
    print("means", g0.mean(), g1.mean())
    # t-test
    t, p = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
    print("t", t, "p", p)
    # effect size Cohen's d
    def cohend(a,b):
        na, nb = len(a), len(b)
        va, vb = a.var(ddof=1), b.var(ddof=1)
        s = np.sqrt(((na-1)*va + (nb-1)*vb)/(na+nb-2))
        return (a.mean()-b.mean())/s
    if len(g0)>1 and len(g1)>1:
        print("d", cohend(g1,g0))

