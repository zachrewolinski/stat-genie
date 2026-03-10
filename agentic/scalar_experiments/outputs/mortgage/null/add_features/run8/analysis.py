import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Define variables
outcome = "accept"
gender = "female"
# Use core mortgage/credit variables as controls
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

use_cols = [outcome, gender] + covariates

df_sub = df[use_cols].dropna().copy()

# Ensure binary is numeric
for col in [outcome, gender, "black", "self_employed", "married", "bad_history", "denied_PMI"]:
    df_sub[col] = pd.to_numeric(df_sub[col])

# Unadjusted acceptance rates by gender
accept_rate = df_sub.groupby(gender)[outcome].mean()
counts = df_sub.groupby(gender)[outcome].sum()
ns = df_sub.groupby(gender)[outcome].count()

# Two-proportion z-test (female=1 vs female=0)
# Order by gender values to keep interpretation clear
counts = counts.sort_index()
ns = ns.sort_index()
stat, pval_unadj = proportions_ztest(counts.values, ns.values)

# Logistic regression with controls
formula = outcome + " ~ " + gender + " + " + " + ".join(covariates)
model = smf.logit(formula, data=df_sub).fit(disp=False)
coef = model.params[gender]
pval_adj = model.pvalues[gender]
odds_ratio = float(np.exp(coef))

# Average marginal effect of gender
margeff = model.get_margeff(at="overall")
marg_df = margeff.summary_frame()
me = float(marg_df.loc[gender, "dy/dx"])
# Robustly locate p-value column across statsmodels versions
p_col = None
for col in marg_df.columns:
    col_lower = col.lower()
    if "p>|" in col_lower or col_lower in {"pvalue", "p-value", "p"} or "pr(>|" in col_lower:
        p_col = col
        break
if p_col is not None:
    me_p = float(marg_df.loc[gender, p_col])
else:
    # Fallback: compute from z or t statistic if present
    if "z" in marg_df.columns:
        from scipy.stats import norm

        z = float(marg_df.loc[gender, "z"])
        me_p = float(2 * (1 - norm.cdf(abs(z))))
    elif "t" in marg_df.columns:
        from scipy.stats import norm

        t = float(marg_df.loc[gender, "t"])
        me_p = float(2 * (1 - norm.cdf(abs(t))))
    else:
        me_p = float("nan")

# Build explanation
female_accept = float(accept_rate.loc[1.0]) if 1.0 in accept_rate.index else float("nan")
male_accept = float(accept_rate.loc[0.0]) if 0.0 in accept_rate.index else float("nan")

explanation = (
    f"Analyzed {len(df_sub)} applications with non-missing gender, approval, and core credit controls. "
    f"Unadjusted acceptance rates: female={female_accept:.3f}, male={male_accept:.3f}. "
    f"Two-proportion z-test p-value={pval_unadj:.4g}. "
    f"Logistic regression controlling for credit and underwriting factors shows female coefficient={coef:.3f} "
    f"(odds ratio={odds_ratio:.3f}, p-value={pval_adj:.4g}). "
    f"Average marginal effect of female on approval probability={me:.3f} (p-value={me_p:.4g})."
)

# Determine Likert response
# Rule of thumb: if adjusted p-value < 0.05 and effect size is meaningful, say Yes.
# Use marginal effect magnitude for strength.
if pval_adj < 0.05:
    # Scale effect size to Likert: small (<2pp) -> around 60, moderate (2-5pp) -> 70, large (>5pp) -> 80+
    abs_me = abs(me)
    if abs_me < 0.02:
        response = 60
    elif abs_me < 0.05:
        response = 72
    else:
        response = 85
    # Direction: if negative effect for female, still "Yes" (gender affects) but explanation should note direction
else:
    # No significant evidence; keep near 40-50 depending on effect size
    abs_me = abs(me)
    if abs_me < 0.01:
        response = 45
    elif abs_me < 0.03:
        response = 40
    else:
        response = 35

# Write conclusion
conclusion = {"response": int(response), "explanation": explanation}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(conclusion, f)
