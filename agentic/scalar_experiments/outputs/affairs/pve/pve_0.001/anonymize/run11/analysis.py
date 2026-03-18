import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("affairs.csv")

# Map children indicator
_df["children"] = _df["feature6"].map({"yes": 1, "no": 0})

# Affair frequency
_df["affair_freq"] = _df["feature2"].astype(float)

# Binary indicator: any affair (using > 0 threshold)
_df["any_affair"] = (_df["affair_freq"] > 0).astype(int)

# Basic group stats
summary = _df.groupby("children")["affair_freq"].agg(["count", "mean", "median", "std"]).reset_index()

# T-test (Welch)
no_group = _df.loc[_df["children"] == 0, "affair_freq"].dropna()
yes_group = _df.loc[_df["children"] == 1, "affair_freq"].dropna()

t_stat, t_p = stats.ttest_ind(no_group, yes_group, equal_var=False)

# Mann-Whitney U (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(no_group, yes_group, alternative="two-sided")
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Cohen's d
mean_diff = no_group.mean() - yes_group.mean()
pooled_sd = np.sqrt(((no_group.var(ddof=1) + yes_group.var(ddof=1)) / 2))
cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

# Linear regression with controls
# Controls: gender (feature3), age (feature4), years married (feature5),
# religiousness (feature7), education (feature8), occupation (feature9), marriage rating (feature10)
# Use robust standard errors

model_ols = smf.ols(
    "affair_freq ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=_df,
).fit(cov_type="HC3")

# Logistic regression for any affair
model_logit = smf.logit(
    "any_affair ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=_df,
).fit(disp=False)

# Extract key results
ols_coef = model_ols.params.get("children", np.nan)
ols_p = model_ols.pvalues.get("children", np.nan)

logit_coef = model_logit.params.get("children", np.nan)
logit_p = model_logit.pvalues.get("children", np.nan)

# Odds ratio for logit
odds_ratio = np.exp(logit_coef) if np.isfinite(logit_coef) else np.nan

print("SUMMARY")
print(summary)
print("\nTTEST")
print({"t_stat": t_stat, "p_value": t_p, "mean_diff_no_minus_yes": mean_diff, "cohens_d": cohens_d})
print("\nMANNWHITNEY")
print({"u_stat": u_stat, "p_value": u_p})
print("\nOLS")
print({"coef_children": ols_coef, "p_value": ols_p})
print("\nLOGIT")
print({"coef_children": logit_coef, "p_value": logit_p, "odds_ratio": odds_ratio})

# Also show any_affair rate by children
rate = _df.groupby("children")["any_affair"].mean().reset_index()
print("\nANY_AFFAIR_RATE")
print(rate)
