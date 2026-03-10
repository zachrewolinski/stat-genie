import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Drop unnamed index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure expected columns exist
required_cols = {
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
}
missing_cols = sorted(required_cols - set(df.columns))
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# Drop rows with missing data for relevant columns
analysis_cols = list(required_cols)
clean = df[analysis_cols].dropna().copy()

# Basic rates by gender
rates = clean.groupby("female")["accept"].mean()
count = clean.groupby("female")["accept"].size()

# Chi-square test of independence
ct = pd.crosstab(clean["female"], clean["accept"])
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# Logistic regression with controls
formula = (
    "accept ~ female + black + housing_expense_ratio + self_employed + married + "
    "mortgage_credit + consumer_credit + bad_history + PI_ratio + loan_to_value + denied_PMI"
)
model = smf.logit(formula, data=clean)
result = model.fit(disp=False, cov_type="HC1")

coef_female = result.params["female"]
se_female = result.bse["female"]
p_female = result.pvalues["female"]
odds_ratio = float(np.exp(coef_female))

# Average marginal effect for female
margeff = result.get_margeff(at="overall", method="dydx")
me_table = margeff.summary_frame()
me_female = me_table.loc["female", "dy/dx"]
me_female_se = me_table.loc["female", "Std. Err."]
# Statsmodels versions differ on p-value column naming
pval_col = None
for candidate in ["P>|z|", "Pr(>|z|)", "P>|t|", "Pr(>|t|)"]:
    if candidate in me_table.columns:
        pval_col = candidate
        break
if pval_col is None:
    raise KeyError(f"P-value column not found in marginal effects table: {list(me_table.columns)}")
me_female_p = me_table.loc["female", pval_col]

output = {
    "n": int(clean.shape[0]),
    "accept_rate_female": float(rates.get(1.0, np.nan)),
    "accept_rate_male": float(rates.get(0.0, np.nan)),
    "count_female": int(count.get(1.0, 0)),
    "count_male": int(count.get(0.0, 0)),
    "chi2_p": float(p_chi),
    "logit_coef_female": float(coef_female),
    "logit_se_female": float(se_female),
    "logit_p_female": float(p_female),
    "logit_odds_ratio_female": float(odds_ratio),
    "margeff_female": float(me_female),
    "margeff_female_se": float(me_female_se),
    "margeff_female_p": float(me_female_p),
}

print(json.dumps(output, indent=2))
