import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import Table


def chi_square_test(table: pd.DataFrame):
    arr = table.values
    res = Table(arr).test_nominal_association()
    stat = float(res.statistic)
    p = float(res.pvalue)
    n = arr.sum()
    r, c = arr.shape
    dof = (r - 1) * (c - 1)
    denom = n * (min(r, c) - 1) if min(r, c) > 1 else np.nan
    cramer_v = np.sqrt(stat / denom) if denom and denom > 0 else np.nan
    return {"chi2": stat, "p": p, "dof": dof, "n": int(n), "cramer_v": float(cramer_v)}


def format_test(name, res):
    return (
        f"{name}: chi2={res['chi2']:.3f}, dof={res['dof']}, p={res['p']:.4g}, "
        f"cramer_v={res['cramer_v']:.3f}, n={res['n']}"
    )


df = pd.read_csv("boxes.csv")

# Outcome labels
outcome_map = {1: "unchosen", 2: "majority", 3: "minority"}

df["outcome"] = df["y"].map(outcome_map)

# Age groups for developmental stages
bins = [4, 7, 10, 13, 15]
labels = ["4-6", "7-9", "10-12", "13-14"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)

# Social reliance: choosing either demonstrated option (majority or minority)
df["social_choice"] = df["y"].isin([2, 3]).astype(int)

# Majority preference among social choices
social_df = df[df["social_choice"] == 1].copy()
social_df["majority_choice"] = (social_df["y"] == 2).astype(int)

# Contingency tables and tests
results = []

# Full outcome distribution vs culture and age_group
ct_culture_outcome = pd.crosstab(df["culture"], df["outcome"])
ct_age_outcome = pd.crosstab(df["age_group"], df["outcome"])
results.append(format_test("Outcome vs culture", chi_square_test(ct_culture_outcome)))
results.append(format_test("Outcome vs age_group", chi_square_test(ct_age_outcome)))

# Social reliance (social_choice) vs culture and age_group
ct_culture_social = pd.crosstab(df["culture"], df["social_choice"])
ct_age_social = pd.crosstab(df["age_group"], df["social_choice"])
results.append(format_test("Social reliance vs culture", chi_square_test(ct_culture_social)))
results.append(format_test("Social reliance vs age_group", chi_square_test(ct_age_social)))

# Majority preference among social choices vs culture and age_group
ct_culture_majority = pd.crosstab(social_df["culture"], social_df["majority_choice"])
ct_age_majority = pd.crosstab(social_df["age_group"], social_df["majority_choice"])
results.append(format_test("Majority preference vs culture", chi_square_test(ct_culture_majority)))
results.append(format_test("Majority preference vs age_group", chi_square_test(ct_age_majority)))

# Rates by culture and age_group
social_rate_by_culture = df.groupby("culture")["social_choice"].mean()
majority_rate_by_culture = social_df.groupby("culture")["majority_choice"].mean()

social_rate_by_age = df.groupby("age_group")["social_choice"].mean()
majority_rate_by_age = social_df.groupby("age_group")["majority_choice"].mean()

print("Counts:")
print(df.shape)
print()
print("Social reliance rate by culture:")
print(social_rate_by_culture)
print()
print("Majority preference rate by culture (among social choices):")
print(majority_rate_by_culture)
print()
print("Social reliance rate by age_group:")
print(social_rate_by_age)
print()
print("Majority preference rate by age_group (among social choices):")
print(majority_rate_by_age)
print()
print("Chi-square tests:")
for line in results:
    print(line)
