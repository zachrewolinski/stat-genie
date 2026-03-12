import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "mortgage.csv"
df = pd.read_csv(csv_path)

# Basic clean: drop rows with missing in key columns
# Define outcome as accept (1 accepted) from dataset; if missing use deny.
if "accept" in df.columns:
    outcome_col = "accept"
elif "deny" in df.columns:
    outcome_col = "deny"
else:
    raise ValueError("No accept/deny column found")

# Ensure binary numeric

# Minimal subset for analysis
predictor = "female"

# Check missing
cols_basic = [outcome_col, predictor]

# For adjusted model, include common creditworthiness controls
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

# Keep only columns present
controls_present = [c for c in control_cols if c in df.columns]

# Drop missing for each model separately

df_basic = df[cols_basic].dropna()

# Compute approval rates by gender
basic_rates = df_basic.groupby(predictor)[outcome_col].agg(['mean','count'])

# Unadjusted logistic regression
# Note: if outcome is accept, logistic regression for accept=1; if outcome is deny, then deny=1.
formula_basic = f"{outcome_col} ~ {predictor}"
model_basic = smf.logit(formula_basic, data=df_basic).fit(disp=False)

# Adjusted logistic regression
cols_adj = [outcome_col, predictor] + controls_present

df_adj = df[cols_adj].dropna()

# Build formula
formula_adj = outcome_col + " ~ " + " + ".join([predictor] + controls_present)
model_adj = smf.logit(formula_adj, data=df_adj).fit(disp=False)

# Extract coefficient, p-value, odds ratio
coef_basic = model_basic.params[predictor]
pval_basic = model_basic.pvalues[predictor]
or_basic = float(np.exp(coef_basic))

coef_adj = model_adj.params[predictor]
pval_adj = model_adj.pvalues[predictor]
or_adj = float(np.exp(coef_adj))

# Confidence intervals for odds ratio
ci_basic = model_basic.conf_int().loc[predictor].tolist()
ci_adj = model_adj.conf_int().loc[predictor].tolist()
ci_or_basic = [float(np.exp(ci_basic[0])), float(np.exp(ci_basic[1]))]
ci_or_adj = [float(np.exp(ci_adj[0])), float(np.exp(ci_adj[1]))]

results = {
    "outcome": outcome_col,
    "n_basic": int(df_basic.shape[0]),
    "n_adj": int(df_adj.shape[0]),
    "approval_rates_by_female": basic_rates.reset_index().to_dict(orient="records"),
    "basic_logit": {
        "coef_female": float(coef_basic),
        "pval_female": float(pval_basic),
        "odds_ratio_female": or_basic,
        "odds_ratio_ci": ci_or_basic,
    },
    "adjusted_logit": {
        "coef_female": float(coef_adj),
        "pval_female": float(pval_adj),
        "odds_ratio_female": or_adj,
        "odds_ratio_ci": ci_or_adj,
        "controls": controls_present,
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
