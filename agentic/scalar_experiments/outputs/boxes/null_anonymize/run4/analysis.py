import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load data
path = "boxes.csv"
df = pd.read_csv(path)

# Define outcomes
# feature1: 1=unchosen, 2=majority, 3=minority

df["social_choice"] = df["feature1"].isin([2, 3]).astype(int)
df["majority_choice"] = (df["feature1"] == 2).astype(int)

# Age groups as developmental stages
bins = [3.5, 6.5, 9.5, 12.5, 14.5]
labels = ["4-6", "7-9", "10-12", "13-14"]
df["age_group"] = pd.cut(df["feature3"], bins=bins, labels=labels, include_lowest=True)

results = []

def chi_square_test(var, outcome, data):
    table = pd.crosstab(data[var], data[outcome])
    chi2, p, dof, expected = chi2_contingency(table)
    # Cramer's V
    n = table.to_numpy().sum()
    r, k = table.shape
    v = np.sqrt(chi2 / (n * (min(r-1, k-1)))) if min(r-1, k-1) > 0 else np.nan
    return p, v, table

# Tests for reliance on social information (social_choice)
for var in ["feature5", "age_group"]:
    p, v, table = chi_square_test(var, "social_choice", df)
    results.append({"measure": "social_choice", "var": var, "p": p, "v": v, "table": table})

# Tests for preference for majority cues among social choosers
social_df = df[df["social_choice"] == 1]
for var in ["feature5", "age_group"]:
    p, v, table = chi_square_test(var, "majority_choice", social_df)
    results.append({"measure": "majority_choice", "var": var, "p": p, "v": v, "table": table})

# Scoring
score = 0
for res in results:
    p = res["p"]
    v = res["v"]
    if p < 0.001:
        score += 25
    elif p < 0.01:
        score += 20
    elif p < 0.05:
        score += 15
    elif p < 0.1:
        score += 5
    else:
        score -= 10

    if v >= 0.2:
        score += 10
    elif v >= 0.1:
        score += 5
    elif v >= 0.05:
        score += 2

score = int(max(-100, min(100, round(score))))

# Print summary for inspection
print("Results:")
for res in results:
    print(res["measure"], res["var"], "p=%.6f" % res["p"], "V=%.3f" % res["v"])

print("Score:", score)

# Write conclusion
with open("conclusion.txt", "w") as f:
    f.write(str(score))
