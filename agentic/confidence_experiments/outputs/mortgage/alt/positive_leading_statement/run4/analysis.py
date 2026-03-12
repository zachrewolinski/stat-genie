import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Basic cleaning: drop rows with missing in relevant columns
# Identify outcome and predictor columns
outcome = "accept"  # 1 accepted, 0 denied
predictor = "female"

covariates = [
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

cols = [outcome, predictor] + covariates

df_model = df[cols].dropna()

# Descriptive approval rates by gender
rates = df_model.groupby(predictor)[outcome].agg(["mean", "count", "sum"]).rename(columns={"sum": "approved"})

# Two-proportion z-test (difference in approval rates)
count = rates.loc[:, "approved"].values
nobs = rates.loc[:, "count"].values
z_stat, p_val = proportions_ztest(count, nobs)

# Unadjusted logistic regression (female only)
X_unadj = sm.add_constant(df_model[[predictor]])
model_unadj = sm.Logit(df_model[outcome], X_unadj).fit(disp=False)

# Adjusted logistic regression (female + covariates)
X_adj = sm.add_constant(df_model[[predictor] + covariates])
model_adj = sm.Logit(df_model[outcome], X_adj).fit(disp=False)

# Extract coefficient, odds ratio, p-value for female
coef_unadj = model_unadj.params[predictor]
se_unadj = model_unadj.bse[predictor]
p_unadj = model_unadj.pvalues[predictor]

coef_adj = model_adj.params[predictor]
se_adj = model_adj.bse[predictor]
p_adj = model_adj.pvalues[predictor]

or_unadj = float(np.exp(coef_unadj))
or_adj = float(np.exp(coef_adj))

# 95% CI for odds ratios
ci_unadj = model_unadj.conf_int().loc[predictor]
ci_adj = model_adj.conf_int().loc[predictor]

or_ci_unadj = (float(np.exp(ci_unadj[0])), float(np.exp(ci_unadj[1])))
or_ci_adj = (float(np.exp(ci_adj[0])), float(np.exp(ci_adj[1])))

summary = {
    "n_total": int(df_model.shape[0]),
    "rates_by_gender": {
        "male_female_0": {
            "approval_rate": float(rates.loc[0, "mean"]) if 0 in rates.index else None,
            "n": int(rates.loc[0, "count"]) if 0 in rates.index else None,
        },
        "female_1": {
            "approval_rate": float(rates.loc[1, "mean"]) if 1 in rates.index else None,
            "n": int(rates.loc[1, "count"]) if 1 in rates.index else None,
        },
    },
    "two_proportion_test": {
        "z": float(z_stat),
        "p": float(p_val),
    },
    "logit_unadjusted": {
        "coef": float(coef_unadj),
        "se": float(se_unadj),
        "p": float(p_unadj),
        "odds_ratio": float(or_unadj),
        "or_ci95": [or_ci_unadj[0], or_ci_unadj[1]],
    },
    "logit_adjusted": {
        "coef": float(coef_adj),
        "se": float(se_adj),
        "p": float(p_adj),
        "odds_ratio": float(or_adj),
        "or_ci95": [or_ci_adj[0], or_ci_adj[1]],
    },
}

print(json.dumps(summary, indent=2))
