import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Mapping based on info.json descriptions:
# 'religiousness' column is described as children indicator (yes/no)
# 'age' column is described as affairs frequency (with possible noise)
children_col = "religiousness"
affairs_col = "age"

# Basic cleaning
# Ensure children is binary yes/no
children = df[children_col].astype(str).str.lower()

# Some datasets might use 'yes'/'no' only; filter accordingly
mask = children.isin(["yes", "no"])

# Ensure affairs variable is numeric
affairs = pd.to_numeric(df[affairs_col], errors="coerce")

# Subset to valid rows
sub = df.loc[mask].copy()
sub["children_bin"] = (children[mask] == "yes").astype(int)
sub["affairs_var"] = affairs[mask]

# Drop NaNs in affairs
sub = sub.dropna(subset=["affairs_var"])

# Group stats
grp = sub.groupby("children_bin")["affairs_var"]
summary = grp.agg(["count", "mean", "median", "std"]).rename(index={0: "no_children", 1: "children"})

# Welch t-test
no_vals = sub.loc[sub["children_bin"] == 0, "affairs_var"]
yes_vals = sub.loc[sub["children_bin"] == 1, "affairs_var"]

t_stat, t_p = stats.ttest_ind(no_vals, yes_vals, equal_var=False, nan_policy="omit")

# Mann-Whitney U (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(no_vals, yes_vals, alternative="two-sided")
except Exception:
    u_stat, u_p = np.nan, np.nan

# Effect size (Cohen's d)
# Use pooled SD with Welch correction
n1, n2 = len(no_vals), len(yes_vals)
std1, std2 = np.nanstd(no_vals, ddof=1), np.nanstd(yes_vals, ddof=1)
pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else np.nan
cohens_d = (np.nanmean(yes_vals) - np.nanmean(no_vals)) / pooled if pooled and not np.isnan(pooled) else np.nan

# Regression with robust SE
X = sm.add_constant(sub["children_bin"])
model = sm.OLS(sub["affairs_var"], X).fit(cov_type="HC3")

print("Summary by children status:")
print(summary)
print("\nWelch t-test: t=", t_stat, "p=", t_p)
print("Mann-Whitney U: U=", u_stat, "p=", u_p)
print("Cohen's d (children - no children):", cohens_d)
print("\nOLS (HC3) coef for children:")
print(model.params)
print("p-value:", model.pvalues)
