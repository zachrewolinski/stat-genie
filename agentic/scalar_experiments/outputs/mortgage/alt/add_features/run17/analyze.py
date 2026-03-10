import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "mortgage.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# Core mortgage-related columns
outcome = "deny"  # 1 denied, 0 accepted
features = [
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
]

# Basic cleanliness
cols = [outcome] + features
sub = df[cols].copy()
sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

# Ensure binary outcome
sub[outcome] = sub[outcome].astype(int)

# Contingency table for raw association
ct = pd.crosstab(sub["female"], sub[outcome])
# Ensure both columns 0/1 exist
ct = ct.reindex(index=[0,1], columns=[0,1], fill_value=0)
chi2, p_chi, dof, expected = stats.chi2_contingency(ct.values)

# Denial/approval rates by gender
rates = (
    sub.groupby("female")[outcome]
    .agg(["mean", "count"])
    .rename(columns={"mean": "deny_rate", "count": "n"})
)
# approval rate = 1 - deny rate
rates["approve_rate"] = 1 - rates["deny_rate"]

# Logistic regression with controls
X = sm.add_constant(sub[features], has_constant="add")
y = sub[outcome]

model = None
try:
    model = sm.Logit(y, X).fit(disp=False)
except Exception:
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

coef = model.params.get("female", np.nan)
se = model.bse.get("female", np.nan)
pval = model.pvalues.get("female", np.nan)

# Odds ratio and CI
or_val = float(np.exp(coef)) if np.isfinite(coef) else np.nan
ci_low = float(np.exp(coef - 1.96 * se)) if np.isfinite(se) and np.isfinite(coef) else np.nan
ci_high = float(np.exp(coef + 1.96 * se)) if np.isfinite(se) and np.isfinite(coef) else np.nan

# Marginal effect (average)
# Average marginal effect on denial probability by toggling female
try:
    X0 = X.copy()
    X1 = X.copy()
    X0["female"] = 0
    X1["female"] = 1
    pred0 = model.predict(X0)
    pred1 = model.predict(X1)
    marg_eff = float((pred1 - pred0).mean())
except Exception:
    marg_eff = np.nan

# Output summary
summary = {
    "n_used": int(sub.shape[0]),
    "chi2_p": float(p_chi),
    "deny_rate_female": float(rates.loc[1, "deny_rate"]) if 1 in rates.index else np.nan,
    "deny_rate_male": float(rates.loc[0, "deny_rate"]) if 0 in rates.index else np.nan,
    "approve_rate_female": float(rates.loc[1, "approve_rate"]) if 1 in rates.index else np.nan,
    "approve_rate_male": float(rates.loc[0, "approve_rate"]) if 0 in rates.index else np.nan,
    "logit_coef_female": float(coef),
    "logit_p_female": float(pval),
    "odds_ratio_female": float(or_val),
    "or_ci_low": float(ci_low),
    "or_ci_high": float(ci_high),
    "avg_marginal_effect_female": float(marg_eff) if np.isfinite(marg_eff) else np.nan,
}

print(json.dumps(summary, indent=2))
