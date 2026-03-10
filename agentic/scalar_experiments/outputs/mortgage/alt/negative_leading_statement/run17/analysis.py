import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "mortgage.csv"


df = pd.read_csv(DATA_PATH)

# Basic cleaning
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure binary columns are numeric 0/1
binary_cols = ["female", "black", "self_employed", "married", "bad_history", "deny", "denied_PMI", "accept"]
for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing in key vars
key_cols = ["accept", "female"]
control_cols = [
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
use_cols = key_cols + [c for c in control_cols if c in df.columns]
analysis_df = df[use_cols].dropna()

# Descriptive rates
rate_by_gender = analysis_df.groupby("female")["accept"].mean()
count_by_gender = analysis_df.groupby("female")["accept"].size()

# Chi-square test of independence
contingency = pd.crosstab(analysis_df["female"], analysis_df["accept"])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression: unadjusted
logit_unadj = smf.logit("accept ~ female", data=analysis_df).fit(disp=False)

# Logistic regression: adjusted
formula = "accept ~ female + " + " + ".join([c for c in control_cols if c in analysis_df.columns])
logit_adj = smf.logit(formula, data=analysis_df).fit(disp=False)

# Extract results
female_unadj_coef = logit_unadj.params.get("female")
female_unadj_p = logit_unadj.pvalues.get("female")

female_adj_coef = logit_adj.params.get("female")
female_adj_p = logit_adj.pvalues.get("female")

# Convert coefficients to odds ratios
female_unadj_or = float(np.exp(female_unadj_coef)) if female_unadj_coef is not None else None
female_adj_or = float(np.exp(female_adj_coef)) if female_adj_coef is not None else None

results = {
    "n": int(len(analysis_df)),
    "accept_rate_female": float(rate_by_gender.get(1.0, np.nan)),
    "accept_rate_male": float(rate_by_gender.get(0.0, np.nan)),
    "count_female": int(count_by_gender.get(1.0, 0)),
    "count_male": int(count_by_gender.get(0.0, 0)),
    "chi2_p": float(p_chi2),
    "logit_unadj_female_coef": float(female_unadj_coef),
    "logit_unadj_female_p": float(female_unadj_p),
    "logit_unadj_female_or": float(female_unadj_or),
    "logit_adj_female_coef": float(female_adj_coef),
    "logit_adj_female_p": float(female_adj_p),
    "logit_adj_female_or": float(female_adj_or),
}

print(json.dumps(results, indent=2))
