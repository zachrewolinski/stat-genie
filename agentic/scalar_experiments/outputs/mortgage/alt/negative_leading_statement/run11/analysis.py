import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning: drop rows with missing values in relevant columns
outcome = "accept"
key_var = "female"
control_vars = [
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

cols = [outcome, key_var] + control_vars

df_sub = df[cols].dropna()

# Unadjusted acceptance rates by gender
rates = df_sub.groupby(key_var)[outcome].mean().rename({0: "male", 1: "female"})
counts = df_sub.groupby(key_var)[outcome].agg(["sum", "count"]).rename(index={0: "male", 1: "female"})

# Two-proportion z-test (female vs male)
# success = accepted
success = counts["sum"].values
nobs = counts["count"].values

# statsmodels proportion ztest expects array
z_stat, p_val = sm.stats.proportions_ztest(success, nobs)

# Logistic regression with controls
X = df_sub[[key_var] + control_vars]
X = sm.add_constant(X)

y = df_sub[outcome]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

coef = result.params[key_var]
se = result.bse[key_var]
odds_ratio = float(np.exp(coef))

# 95% CI for odds ratio
ci_low, ci_high = result.conf_int().loc[key_var]
ci_low_or = float(np.exp(ci_low))
ci_high_or = float(np.exp(ci_high))

p_val_coef = result.pvalues[key_var]

output = {
    "n": int(len(df_sub)),
    "accept_rate_male": float(rates.loc["male"]),
    "accept_rate_female": float(rates.loc["female"]),
    "accept_rate_diff_female_minus_male": float(rates.loc["female"] - rates.loc["male"]),
    "unadjusted_z": float(z_stat),
    "unadjusted_p": float(p_val),
    "logit_coef_female": float(coef),
    "logit_se_female": float(se),
    "logit_p_female": float(p_val_coef),
    "odds_ratio_female": odds_ratio,
    "odds_ratio_95ci": [ci_low_or, ci_high_or],
}

print(json.dumps(output, indent=2))
