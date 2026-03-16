import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv("mortgage.csv")

# Keep only rows with non-missing values for variables used below
analysis_cols = [
    "feature2",  # female
    "feature14",  # accepted
    "feature3",
    "feature4",
    "feature5",
    "feature6",
    "feature7",
    "feature8",
    "feature9",
    "feature10",
    "feature12",
    "feature13",
]
df = df[analysis_cols].copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Define key variables
female = df["feature2"]
accepted = df["feature14"]

# Basic group stats
group = df.groupby(female)["feature14"].agg(["mean", "sum", "count"]).rename(index={0: "male", 1: "female"})

# Chi-square test for association
contingency = pd.crosstab(female, accepted)
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Two-proportion z-test (female - male)
# proportions: accepted rate by gender
p_f = group.loc["female", "mean"]
p_m = group.loc["male", "mean"]

n_f = group.loc["female", "count"]
n_m = group.loc["male", "count"]

p_pool = (group.loc["female", "sum"] + group.loc["male", "sum"]) / (n_f + n_m)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_f + 1 / n_m))
if se == 0:
    z = np.nan
    p_z = np.nan
else:
    z = (p_f - p_m) / se
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))

# 95% CI for difference in proportions (unpooled)
se_unpooled = np.sqrt(p_f * (1 - p_f) / n_f + p_m * (1 - p_m) / n_m)
ci_low = (p_f - p_m) - 1.96 * se_unpooled
ci_high = (p_f - p_m) + 1.96 * se_unpooled

# Logistic regression controlling for other features (excluding ID-like feature1 and redundant target feature11)
features = [
    "feature2",  # female
    "feature3",
    "feature4",
    "feature5",
    "feature6",
    "feature7",
    "feature8",
    "feature9",
    "feature10",
    "feature12",
    "feature13",
]

X = df[features].copy()
X = sm.add_constant(X, has_constant="add")
model = sm.Logit(accepted, X).fit(disp=0)
coef_female = model.params["feature2"]
se_female = model.bse["feature2"]
p_female = model.pvalues["feature2"]

# Odds ratio and 95% CI for female effect
or_female = float(np.exp(coef_female))
ci_or_low = float(np.exp(coef_female - 1.96 * se_female))
ci_or_high = float(np.exp(coef_female + 1.96 * se_female))

summary = {
    "group_acceptance": group.to_dict(),
    "chi2": float(chi2),
    "p_chi2": float(p_chi2),
    "z_two_prop": float(z) if np.isfinite(z) else None,
    "p_two_prop": float(p_z) if np.isfinite(p_z) else None,
    "diff_prop": float(p_f - p_m),
    "diff_prop_ci": [float(ci_low), float(ci_high)],
    "logit_coef_female": float(coef_female),
    "logit_or_female": or_female,
    "logit_or_ci": [ci_or_low, ci_or_high],
    "logit_p_female": float(p_female),
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
