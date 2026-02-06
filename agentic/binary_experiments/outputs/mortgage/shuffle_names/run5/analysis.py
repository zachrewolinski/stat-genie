import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
df = pd.read_csv('mortgage.csv')

# Per metadata, `denied_PMI` is actually the gender indicator (1=female, 0=male)
# and `deny` appears to encode approval (mean ~0.88).
GENDER_COL = 'denied_PMI'
APPROVAL_COL = 'deny'

use = df[[GENDER_COL, APPROVAL_COL]].copy()
use = use.dropna(subset=[GENDER_COL, APPROVAL_COL])

female = use[GENDER_COL].astype(int)
approval = use[APPROVAL_COL].astype(int)

# Basic rates
rate_female = approval[female == 1].mean()
rate_male = approval[female == 0].mean()
count_female = int((female == 1).sum())
count_male = int((female == 0).sum())

# Two-proportion z-test
counts = np.array([approval[female == 1].sum(), approval[female == 0].sum()])
ns = np.array([count_female, count_male])
stat, pval = proportions_ztest(counts, ns)

# Logistic regression: approval ~ female
X = sm.add_constant(female)
model = sm.Logit(approval, X).fit(disp=False)
coef = model.params[GENDER_COL]
pval_logit = model.pvalues[GENDER_COL]
odds_ratio = np.exp(coef)

# Save key results
results = {
    'count_female': count_female,
    'count_male': count_male,
    'approval_rate_female': float(rate_female),
    'approval_rate_male': float(rate_male),
    'rate_diff_female_minus_male': float(rate_female - rate_male),
    'pval_ztest': float(pval),
    'logit_coef_female': float(coef),
    'logit_odds_ratio_female': float(odds_ratio),
    'logit_pval_female': float(pval_logit),
}

pd.Series(results).to_csv('analysis_results.csv')

print('Counts (female, male):', count_female, count_male)
print('Approval rate female:', rate_female)
print('Approval rate male:', rate_male)
print('Rate diff (female - male):', rate_female - rate_male)
print('Two-proportion z-test p-value:', pval)
print('Logit coef (female):', coef)
print('Logit OR (female):', odds_ratio)
print('Logit p-value:', pval_logit)
