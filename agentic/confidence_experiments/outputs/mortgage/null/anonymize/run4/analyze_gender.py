import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = "mortgage.csv"
df = pd.read_csv(csv_path)

female = "feature2"   # 1 if female
accept = "feature14"  # 1 if accepted

df = df.copy()

# Drop rows with missing values in any relevant columns
# Use all columns except feature11 (redundant denial indicator)
analysis_cols = [c for c in df.columns if c != 'feature11']
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=analysis_cols)

# Basic counts
n_total = len(df)

# Acceptance rates by gender
rates = df.groupby(female)[accept].agg(['mean', 'count'])
rate_female = rates.loc[1, 'mean'] if 1 in rates.index else np.nan
rate_male = rates.loc[0, 'mean'] if 0 in rates.index else np.nan
rate_diff = rate_female - rate_male
count_f = rates.loc[1, 'count'] if 1 in rates.index else 0
count_m = rates.loc[0, 'count'] if 0 in rates.index else 0
succ_f = df.loc[df[female] == 1, accept].sum()
succ_m = df.loc[df[female] == 0, accept].sum()

# Two-proportion z-test
if count_f > 0 and count_m > 0:
    p_pool = (succ_f + succ_m) / (count_f + count_m)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / count_f + 1 / count_m))
    z = (rate_female - rate_male) / se if se > 0 else np.nan
    pval_diff = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
else:
    z = np.nan
    pval_diff = np.nan

# Logistic regression with controls
control_cols = [c for c in df.columns if c not in [accept, 'feature11']]
X = df[control_cols]
X = sm.add_constant(X, has_constant='add')

model = sm.Logit(df[accept], X)
res = model.fit(disp=False)

coef_female = res.params.get(female, np.nan)
pval_female = res.pvalues.get(female, np.nan)
odds_ratio_female = np.exp(coef_female) if np.isfinite(coef_female) else np.nan

# Marginal effect at mean
try:
    margeff = res.get_margeff(at='mean', method='dydx')
    me_table = margeff.summary_frame()
    me_female = me_table.loc[female, 'dy/dx'] if female in me_table.index else np.nan
    me_pval = me_table.loc[female, 'Pr(>|z|)'] if female in me_table.index else np.nan
except Exception:
    me_female = np.nan
    me_pval = np.nan

output = {
    "n_total": int(n_total),
    "accept_rate_male": float(rate_male),
    "accept_rate_female": float(rate_female),
    "rate_diff_female_minus_male": float(rate_diff),
    "rate_diff_pval": float(pval_diff),
    "logit_coef_female": float(coef_female),
    "logit_pval_female": float(pval_female),
    "logit_odds_ratio_female": float(odds_ratio_female),
    "marginal_effect_female": float(me_female),
    "marginal_effect_pval": float(me_pval)
}

print(json.dumps(output, indent=2))
