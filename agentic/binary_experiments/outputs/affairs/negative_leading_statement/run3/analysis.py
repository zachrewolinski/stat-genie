import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

# Load data
df = pd.read_csv("affairs.csv")

# Basic derived columns
df["any_affair"] = (df["affairs"] > 0).astype(int)

# Descriptive stats by children
group = df.groupby("children", dropna=False)
desc = group["affairs"].agg(["mean", "median", "count"])
prop_any = group["any_affair"].mean().rename("prop_any")

# Two-sample t-test for mean affairs (children yes vs no)
yes = df[df["children"] == "yes"]["affairs"]
no = df[df["children"] == "no"]["affairs"]
t_stat, p_value, _ = ttest_ind(yes, no, usevar="unequal")

# Proportion test for any affair
count = np.array([
    df[df["children"] == "yes"]["any_affair"].sum(),
    df[df["children"] == "no"]["any_affair"].sum(),
])
nobs = np.array([
    df[df["children"] == "yes"]["any_affair"].count(),
    df[df["children"] == "no"]["any_affair"].count(),
])
z_stat, p_prop = proportions_ztest(count, nobs)

# Regression models
# OLS on affairs with controls
ols = smf.ols(
    "affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
    data=df,
).fit()

# Logistic regression on any affair
logit = smf.logit(
    "any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating",
    data=df,
).fit(disp=False)

# Extract child effect
ols_coef = ols.params.get("C(children)[T.yes]", np.nan)
ols_p = ols.pvalues.get("C(children)[T.yes]", np.nan)
logit_coef = logit.params.get("C(children)[T.yes]", np.nan)
logit_p = logit.pvalues.get("C(children)[T.yes]", np.nan)
logit_or = float(np.exp(logit_coef)) if pd.notnull(logit_coef) else np.nan

print("Descriptive stats (affairs) by children:\n", desc)
print("\nProportion any affair by children:\n", prop_any)
print(f"\nT-test (mean affairs yes vs no): t={t_stat:.3f}, p={p_value:.3f}")
print(f"Proportion z-test (any affair yes vs no): z={z_stat:.3f}, p={p_prop:.3f}")
print("\nOLS children coef:", ols_coef, "p=", ols_p)
print("Logit children coef:", logit_coef, "p=", logit_p, "odds ratio=", logit_or)
