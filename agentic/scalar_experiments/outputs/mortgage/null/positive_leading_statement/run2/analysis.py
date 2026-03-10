import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Define outcome and key predictor
outcome = "accept"
predictor = "female"

# Basic cleaning: keep rows with non-missing outcome and predictor
base_cols = [outcome, predictor]
base = df[base_cols].dropna()

# Proportion test / chi-square for independence
cont_table = pd.crosstab(base[predictor], base[outcome])
chi2, p_chi, dof, expected = stats.chi2_contingency(cont_table)

# Difference in approval rates (female=1 vs male=0)
rate_female = base.loc[base[predictor] == 1, outcome].mean()
rate_male = base.loc[base[predictor] == 0, outcome].mean()
rate_diff = rate_female - rate_male

# Two-proportion z-test (approx) for difference in approval rates
n_f = (base[predictor] == 1).sum()
n_m = (base[predictor] == 0).sum()
# pooled proportion
p_pool = base[outcome].mean()
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_f + 1/n_m))
if se > 0:
    z = rate_diff / se
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_z = np.nan

# Logistic regression with controls
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

model_cols = [outcome, predictor] + control_vars
model_df = df[model_cols].dropna()

X = model_df[[predictor] + control_vars]
X = sm.add_constant(X, has_constant="add")
y = model_df[outcome]

logit = sm.Logit(y, X)
result = logit.fit(disp=False)

coef = result.params[predictor]
se_coef = result.bse[predictor]
p_value = result.pvalues[predictor]
odds_ratio = float(np.exp(coef))

output = {
    "n_total": int(len(df)),
    "n_base": int(len(base)),
    "approval_rate_female": float(rate_female),
    "approval_rate_male": float(rate_male),
    "approval_rate_diff": float(rate_diff),
    "chi2_p": float(p_chi),
    "z_p": float(p_z),
    "logit_coef_female": float(coef),
    "logit_se_female": float(se_coef),
    "logit_p_female": float(p_value),
    "logit_odds_ratio_female": float(odds_ratio),
    "n_logit": int(len(model_df)),
}

with open("analysis_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
