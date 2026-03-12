import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning: drop rows with missing values in relevant columns
cols = [
    "female",
    "deny",
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

df = df[cols].copy()

# Drop missing
n_before = len(df)
df = df.dropna()

# Denial rates by gender
rate_by_gender = df.groupby("female")["deny"].mean().to_dict()
counts_by_gender = df.groupby("female")["deny"].agg(["count", "sum"]).to_dict(orient="index")

# Chi-square test for independence
cont = pd.crosstab(df["female"], df["deny"])
chi2, p_chi, dof, expected = stats.chi2_contingency(cont)

# Difference in proportions (female - male)
# female=1, male=0
n_f = counts_by_gender.get(1.0, {}).get("count", 0)
ny_f = counts_by_gender.get(1.0, {}).get("sum", 0)

n_m = counts_by_gender.get(0.0, {}).get("count", 0)
ny_m = counts_by_gender.get(0.0, {}).get("sum", 0)

p_f = ny_f / n_f if n_f > 0 else np.nan
p_m = ny_m / n_m if n_m > 0 else np.nan

diff = p_f - p_m

# 95% CI for difference in proportions (Wald)
se_diff = np.sqrt(p_f * (1 - p_f) / n_f + p_m * (1 - p_m) / n_m) if n_f > 0 and n_m > 0 else np.nan
ci_low = diff - 1.96 * se_diff
ci_high = diff + 1.96 * se_diff

# Logistic regression: unadjusted (HC1 robust SEs)
X_simple = sm.add_constant(df[["female"]])
model_simple = sm.Logit(df["deny"], X_simple).fit(disp=False, cov_type="HC1")

# Logistic regression: adjusted for covariates (HC1 robust SEs)
X_full = sm.add_constant(
    df[
        [
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
    ]
)
model_full = sm.Logit(df["deny"], X_full).fit(disp=False, cov_type="HC1")

# Extract female coefficient info

def coef_info(res, name="female"):
    coef = res.params[name]
    se = res.bse[name]
    pval = res.pvalues[name]
    odds = np.exp(coef)
    return {
        "coef": float(coef),
        "se": float(se),
        "pval": float(pval),
        "odds_ratio": float(odds),
    }

out = {
    "n_rows_before": int(n_before),
    "n_rows_after": int(len(df)),
    "denial_rate_by_gender": {"male": float(p_m), "female": float(p_f)},
    "counts_by_gender": {
        "male": {"n": int(n_m), "denied": int(ny_m)},
        "female": {"n": int(n_f), "denied": int(ny_f)},
    },
    "diff_female_minus_male": float(diff),
    "diff_ci_95": [float(ci_low), float(ci_high)],
    "chi_square": {"chi2": float(chi2), "p_value": float(p_chi), "dof": int(dof)},
    "logit_simple_hc1": coef_info(model_simple),
    "logit_full_hc1": coef_info(model_full),
}

print(json.dumps(out, indent=2))
