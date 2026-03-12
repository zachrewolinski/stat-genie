import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Drop rows with missing values in key columns
key_cols = [
    "female",
    "accept",
    "black",
    "housing_expense_ratio",
    "self_employed",
    "married",
    "mortgage_credit",
    "consumer_credit",
    "bad_history",
    "PI_ratio",
    "loan_to_value",
    "denied_PMI",
]

# Ensure columns exist
missing = [c for c in key_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

analysis_df = df[key_cols].dropna().copy()

# Descriptive stats
n_total = len(analysis_df)
counts = analysis_df["female"].value_counts().sort_index()

# Approval rates by gender
rate_by_gender = analysis_df.groupby("female")["accept"].mean()

# Contingency table for chi-square: female (rows) x accept (cols)
cont_table = pd.crosstab(analysis_df["female"], analysis_df["accept"])
chi2, chi2_p, dof, expected = stats.chi2_contingency(cont_table)

# Difference in proportions (female - male)
rate_female = rate_by_gender.loc[1] if 1 in rate_by_gender.index else np.nan
rate_male = rate_by_gender.loc[0] if 0 in rate_by_gender.index else np.nan
rate_diff = rate_female - rate_male

# Unadjusted logistic regression
X_unadj = sm.add_constant(analysis_df[["female"]])
y = analysis_df["accept"]
model_unadj = sm.GLM(y, X_unadj, family=sm.families.Binomial())
res_unadj = model_unadj.fit()

# Adjusted logistic regression
covariates = [
    "female",
    "black",
    "housing_expense_ratio",
    "self_employed",
    "married",
    "mortgage_credit",
    "consumer_credit",
    "bad_history",
    "PI_ratio",
    "loan_to_value",
    "denied_PMI",
]
X_adj = sm.add_constant(analysis_df[covariates])
model_adj = sm.GLM(y, X_adj, family=sm.families.Binomial())
res_adj = model_adj.fit()


def extract_effect(res, var):
    coef = res.params[var]
    se = res.bse[var]
    p = res.pvalues[var]
    # 95% CI
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    # odds ratios
    or_val = float(np.exp(coef))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))
    return {
        "coef": float(coef),
        "se": float(se),
        "p": float(p),
        "or": or_val,
        "or_ci_low": or_low,
        "or_ci_high": or_high,
    }

unadj_effect = extract_effect(res_unadj, "female")
adj_effect = extract_effect(res_adj, "female")

results = {
    "n_total": int(n_total),
    "counts_female": {"male_0": int(counts.get(0, 0)), "female_1": int(counts.get(1, 0))},
    "approval_rate_male": float(rate_male),
    "approval_rate_female": float(rate_female),
    "approval_rate_diff_female_minus_male": float(rate_diff),
    "chi2_p": float(chi2_p),
    "unadjusted": unadj_effect,
    "adjusted": adj_effect,
}

print(json.dumps(results, indent=2))
