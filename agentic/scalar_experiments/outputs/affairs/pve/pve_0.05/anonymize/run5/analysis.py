import json
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "affairs.csv"

df = pd.read_csv(DATA_PATH)

# Ensure expected columns
required_cols = {"feature2", "feature6"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Clean/prepare
# feature6 is categorical yes/no
children = df["feature6"].astype(str).str.lower().str.strip()
# Keep only yes/no
mask = children.isin(["yes", "no"]) & df["feature2"].notna()
sub = df.loc[mask, ["feature2"]].copy()
sub["children_yes"] = (children[mask] == "yes").astype(int)

# Split groups
x_yes = sub.loc[sub["children_yes"] == 1, "feature2"].astype(float)
x_no = sub.loc[sub["children_yes"] == 0, "feature2"].astype(float)

# Descriptives
n_yes = x_yes.shape[0]
n_no = x_no.shape[0]
mean_yes = x_yes.mean()
mean_no = x_no.mean()
median_yes = x_yes.median()
median_no = x_no.median()
std_yes = x_yes.std(ddof=1)
std_no = x_no.std(ddof=1)

# Difference (yes - no): negative implies fewer affairs with children
mean_diff = mean_yes - mean_no

# Welch t-test
welch_t = stats.ttest_ind(x_yes, x_no, equal_var=False, nan_policy="omit")

# Mann-Whitney U (two-sided)
# Use alternative="two-sided" to test any distribution shift
mw_u = stats.mannwhitneyu(x_yes, x_no, alternative="two-sided")

# Effect size: Cohen's d (using pooled SD with unequal n)
# Use standard pooled SD
pooled_sd = np.sqrt(((n_yes - 1) * std_yes**2 + (n_no - 1) * std_no**2) / (n_yes + n_no - 2))
cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

# Any-affair indicator
any_yes = (x_yes > 0).mean()
any_no = (x_no > 0).mean()
# Difference in proportions
prop_diff = any_yes - any_no

# Two-proportion z-test
# Use statsmodels if available; implement manually to avoid dependency.
p1 = any_yes
p2 = any_no
n1 = n_yes
n2 = n_no
p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
if se > 0:
    z = (p1 - p2) / se
    p_val_prop = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_val_prop = np.nan

summary = {
    "n_yes": int(n_yes),
    "n_no": int(n_no),
    "mean_yes": float(mean_yes),
    "mean_no": float(mean_no),
    "median_yes": float(median_yes),
    "median_no": float(median_no),
    "std_yes": float(std_yes),
    "std_no": float(std_no),
    "mean_diff_yes_minus_no": float(mean_diff),
    "welch_t_stat": float(welch_t.statistic),
    "welch_t_p": float(welch_t.pvalue),
    "mw_u_stat": float(mw_u.statistic),
    "mw_u_p": float(mw_u.pvalue),
    "cohen_d": float(cohen_d),
    "any_affair_yes": float(any_yes),
    "any_affair_no": float(any_no),
    "any_affair_prop_diff": float(prop_diff),
    "prop_z": float(z),
    "prop_p": float(p_val_prop),
}

print(json.dumps(summary, indent=2))
