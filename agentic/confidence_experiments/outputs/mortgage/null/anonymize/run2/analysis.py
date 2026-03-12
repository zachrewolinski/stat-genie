import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


df = pd.read_csv("mortgage.csv")

# Define key columns
col_gender = "feature2"  # 1 if female, 0 if male
col_accepted = "feature14"  # 1 if accepted, 0 if denied

# Basic sanity checks
if col_gender not in df.columns or col_accepted not in df.columns:
    raise ValueError("Expected columns not found in dataset.")

# Contingency table and chi-square test
cont_table = pd.crosstab(df[col_gender], df[col_accepted])
chi2, p_value, dof, expected = stats.chi2_contingency(cont_table)

# Acceptance rates by gender
accept_rate = df.groupby(col_gender)[col_accepted].mean()

# Logistic regression with controls
predictors = [
    col_gender,
    "feature3",  # Black
    "feature4",  # housing expense ratio
    "feature5",  # self-employed
    "feature6",  # married
    "feature7",  # mortgage credit score
    "feature8",  # consumer credit score
    "feature9",  # bad credit
    "feature10", # debt ratio
    "feature12", # loan-to-value
    "feature13", # PMI denied
]

model_df = df[predictors + [col_accepted]].copy()
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

X = model_df[predictors].copy()
X = sm.add_constant(X, has_constant="add")
y = model_df[col_accepted]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False, maxiter=200)

# Extract gender effect
coef = result.params[col_gender]
se = result.bse[col_gender]
pval = result.pvalues[col_gender]
ci_low, ci_high = result.conf_int().loc[col_gender]

odds_ratio = float(np.exp(coef))
or_ci_low = float(np.exp(ci_low))
or_ci_high = float(np.exp(ci_high))

# Package results for manual interpretation
output = {
    "n": int(len(df)),
    "n_model": int(len(model_df)),
    "accept_rate_male": float(accept_rate.get(0.0, np.nan)),
    "accept_rate_female": float(accept_rate.get(1.0, np.nan)),
    "chi2": float(chi2),
    "chi2_p": float(p_value),
    "logit_coef_gender": float(coef),
    "logit_se_gender": float(se),
    "logit_p_gender": float(pval),
    "logit_or_gender": odds_ratio,
    "logit_or_ci_low": or_ci_low,
    "logit_or_ci_high": or_ci_high,
}

print(json.dumps(output, indent=2))
