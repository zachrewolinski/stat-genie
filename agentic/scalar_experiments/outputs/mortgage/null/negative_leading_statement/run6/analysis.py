import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Outcome: approval (accept=1, denied=0)
# Ensure binary

def summarize_by_gender(data):
    # gender: female 1, male 0
    grouped = data.groupby('female')['accept'].agg(['mean', 'count', 'sum'])
    grouped.rename(columns={'mean':'approval_rate','count':'n','sum':'approved'}, inplace=True)
    return grouped

# Basic clean: keep rows with required columns
required_cols = [
    'female','accept','deny','black','housing_expense_ratio','self_employed','married',
    'mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value'
]
# denied_PMI is potentially post-application; exclude from adjusted model

# Coerce to numeric and drop missing
for c in required_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

base = df.dropna(subset=['female','accept'])

# Approval rate by gender
by_gender = summarize_by_gender(base)

# Difference in proportions (female - male)
if 0 in by_gender.index and 1 in by_gender.index:
    p_male = by_gender.loc[0, 'approval_rate']
    p_female = by_gender.loc[1, 'approval_rate']
    n_male = by_gender.loc[0, 'n']
    n_female = by_gender.loc[1, 'n']
    diff = p_female - p_male
    # Wald CI for difference
    se = np.sqrt(p_female*(1-p_female)/n_female + p_male*(1-p_male)/n_male)
    ci_low = diff - 1.96*se
    ci_high = diff + 1.96*se
else:
    diff = np.nan
    ci_low = np.nan
    ci_high = np.nan

# Chi-square test of independence
contingency = pd.crosstab(base['female'], base['accept'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

# Unadjusted logistic regression: accept ~ female
model_unadj = None
unadj_res = None
try:
    X_unadj = sm.add_constant(base[['female']])
    y = base['accept']
    model_unadj = sm.Logit(y, X_unadj, missing='drop')
    unadj_res = model_unadj.fit(disp=False)
except Exception as e:
    unadj_res = None
    unadj_err = str(e)

# Adjusted logistic regression
adj_cols = [
    'female','black','housing_expense_ratio','self_employed','married','mortgage_credit',
    'consumer_credit','bad_history','PI_ratio','loan_to_value'
]
base_adj = df.dropna(subset=adj_cols + ['accept']).copy()

adj_res = None
try:
    X_adj = sm.add_constant(base_adj[adj_cols])
    y_adj = base_adj['accept']
    model_adj = sm.Logit(y_adj, X_adj, missing='drop')
    # robust SE
    adj_res = model_adj.fit(disp=False)
except Exception as e:
    adj_res = None
    adj_err = str(e)

# Extract metrics
results = {
    "n_total": int(base.shape[0]),
    "n_adj": int(base_adj.shape[0]),
    "approval_by_gender": by_gender.to_dict(),
    "diff_female_minus_male": float(diff),
    "diff_ci_low": float(ci_low),
    "diff_ci_high": float(ci_high),
    "chi2_p": float(p_chi)
}

if unadj_res is not None:
    coef = unadj_res.params['female']
    se = unadj_res.bse['female']
    p = unadj_res.pvalues['female']
    or_val = float(np.exp(coef))
    ci = unadj_res.conf_int().loc['female']
    results.update({
        "unadj_coef": float(coef),
        "unadj_se": float(se),
        "unadj_p": float(p),
        "unadj_or": float(or_val),
        "unadj_or_ci_low": float(np.exp(ci[0])),
        "unadj_or_ci_high": float(np.exp(ci[1]))
    })

if adj_res is not None:
    coef = adj_res.params['female']
    se = adj_res.bse['female']
    p = adj_res.pvalues['female']
    or_val = float(np.exp(coef))
    ci = adj_res.conf_int().loc['female']
    results.update({
        "adj_coef": float(coef),
        "adj_se": float(se),
        "adj_p": float(p),
        "adj_or": float(or_val),
        "adj_or_ci_low": float(np.exp(ci[0])),
        "adj_or_ci_high": float(np.exp(ci[1]))
    })

# Save results for inspection
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
