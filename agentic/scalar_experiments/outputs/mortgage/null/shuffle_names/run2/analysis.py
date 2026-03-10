import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

base = "/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/mortgage/null/shuffle_names/run2"

with open(f"{base}/info.json", "r") as f:
    info = json.load(f)

fields = info["data_desc"]["fields"]

# Map description keywords to column names
female_col = None
accept_col = None
deny_col = None
for field in fields:
    desc = (field.get("properties", {}).get("description") or "").lower()
    col = field["column"]
    if "female" in desc:
        female_col = col
    if "accepted" in desc and "denied" in desc:
        # description about accept/deny
        # decide whether it's accept or deny by phrasing
        if "accepted" in desc and "denied" in desc:
            if "accepted, 0 if denied" in desc:
                accept_col = col
            if "denied, 0 if accepted" in desc:
                deny_col = col

print("female_col", female_col)
print("accept_col", accept_col)
print("deny_col", deny_col)

# Load data

df = pd.read_csv(f"{base}/mortgage.csv")

# Basic checks
print("columns", df.columns.tolist())

# Verify binary columns
for c in [female_col, accept_col, deny_col]:
    if c is None:
        continue
    print(c, df[c].value_counts().sort_index().head())

# Determine outcome column
outcome_col = None
if deny_col is not None:
    outcome_col = deny_col
elif accept_col is not None:
    outcome_col = accept_col

# If both accept and deny, check complement
if accept_col is not None and deny_col is not None:
    comp = ((df[accept_col] + df[deny_col]) == 1).mean()
    print("accept+deny complement fraction", comp)

# Use outcome as deny=1, accept=0
if outcome_col is None:
    raise SystemExit("No outcome column found")

# Ensure binary
print("Outcome mean", df[outcome_col].mean())

# Bivariate analysis: approval rate by gender
# If outcome is deny, approval rate = 1 - deny

g = df[female_col]

# Drop missing
mask = g.notna() & df[outcome_col].notna()

df2 = df.loc[mask, [female_col, outcome_col]].copy()

# compute rates
for sex in [0,1]:
    sub = df2[df2[female_col] == sex]
    if len(sub) == 0:
        print("no data for sex", sex)
        continue
    deny_rate = sub[outcome_col].mean()
    approve_rate = 1 - deny_rate if outcome_col == deny_col else sub[outcome_col].mean()
    print("sex", sex, "n", len(sub), "deny_rate", deny_rate, "approve_rate", approve_rate)

# two-proportion z-test on deny rate
# contingency table
ct = pd.crosstab(df2[female_col], df2[outcome_col])
print("contingency table:\n", ct)

# chi-square test
chi2, p, dof, exp = stats.chi2_contingency(ct)
print("chi2", chi2, "p", p)

# Logistic regression: outcome ~ female
X = sm.add_constant(df2[[female_col]])
model = sm.Logit(df2[outcome_col], X)
res = model.fit(disp=False)
print(res.summary())

# Compute odds ratio for female
params = res.params
conf = res.conf_int()

odds_ratio = np.exp(params[female_col])
ci_low, ci_high = np.exp(conf.loc[female_col])
print("odds_ratio", odds_ratio, "CI", ci_low, ci_high)

# Multivariate logistic regression with controls (exclude outcome, accept/deny complement, and female)

cols = df.columns.tolist()
exclude = {outcome_col, female_col}
# also exclude the other accept/deny column if present
if accept_col is not None:
    exclude.add(accept_col)
if deny_col is not None:
    exclude.add(deny_col)

covars = [c for c in cols if c not in exclude]

# Drop columns with zero variance
covars = [c for c in covars if df[c].nunique() > 1]

# build dataset
full = df[[outcome_col, female_col] + covars].dropna()

# remove columns with extremely high correlation to outcome (perfect separation)
# compute correlation with outcome
corrs = full[covars].corrwith(full[outcome_col])
# remove if absolute correlation 0.99 or more
covars_filtered = [c for c in covars if abs(corrs[c]) < 0.99]

Xf = sm.add_constant(full[[female_col] + covars_filtered])

try:
    model_full = sm.Logit(full[outcome_col], Xf)
    res_full = model_full.fit(disp=False, maxiter=200)
    print(res_full.summary())
    params_full = res_full.params
    conf_full = res_full.conf_int()
    or_full = np.exp(params_full[female_col])
    ci_full = np.exp(conf_full.loc[female_col])
    p_full = res_full.pvalues[female_col]
    print("adj odds_ratio", or_full, "CI", ci_full.values, "p", p_full)
except Exception as e:
    print("full model failed", e)

