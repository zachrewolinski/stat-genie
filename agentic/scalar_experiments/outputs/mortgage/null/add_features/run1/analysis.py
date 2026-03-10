import json
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("mortgage.csv")

# Keep mortgage-related columns that are documented
cols = [
    "deny",
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

missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

sub = df[cols].copy()

# Drop rows with any missing values in selected columns
sub = sub.dropna()

# Ensure binary columns are 0/1 numeric
binary_cols = ["deny", "female", "black", "self_employed", "married", "bad_history", "denied_PMI"]
for c in binary_cols:
    sub[c] = sub[c].astype(int)

# Bivariate association: denial rate by gender
ct = pd.crosstab(sub["female"], sub["deny"])
# Ensure columns for 0/1
if 0 not in ct.columns:
    ct[0] = 0
if 1 not in ct.columns:
    ct[1] = 0
ct = ct[[0, 1]]

chi2, pval, dof, expected = chi2_contingency(ct)

denial_rate_female = ct.loc[1, 1] / ct.loc[1].sum() if 1 in ct.index else np.nan
denial_rate_male = ct.loc[0, 1] / ct.loc[0].sum() if 0 in ct.index else np.nan
rate_diff = denial_rate_female - denial_rate_male

# Multivariate logistic regression
formula = (
    "deny ~ female + black + housing_expense_ratio + self_employed + married + "
    "mortgage_credit + consumer_credit + bad_history + PI_ratio + loan_to_value + denied_PMI"
)

model = smf.glm(formula=formula, data=sub, family=sm.families.Binomial())
result = model.fit()

coef = result.params["female"]
se = result.bse["female"]
pvalue = result.pvalues["female"]

odds_ratio = float(np.exp(coef))

# Collect outputs
out = {
    "n": int(len(sub)),
    "denial_rate_female": float(denial_rate_female),
    "denial_rate_male": float(denial_rate_male),
    "rate_diff": float(rate_diff),
    "chi2_pvalue": float(pval),
    "logit_coef_female": float(coef),
    "logit_se_female": float(se),
    "logit_pvalue_female": float(pvalue),
    "odds_ratio_female": float(odds_ratio),
}

print(json.dumps(out, indent=2))
