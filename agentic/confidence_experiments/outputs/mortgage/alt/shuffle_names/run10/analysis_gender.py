import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data

df = pd.read_csv('mortgage.csv')

# Based on info.json descriptions:
# denied_PMI -> female indicator (1 female, 0 male)
# deny -> approval indicator (1 accepted, 0 denied)

gender_col = 'denied_PMI'
approve_col = 'deny'

sub = df[[gender_col, approve_col]].copy()

# Drop missing gender
sub = sub.dropna()

# Basic counts
counts = sub.groupby(gender_col)[approve_col].agg(['count','mean','sum'])
print('Approval rates by gender (female=1):')
print(counts)

# Contingency table
ct = pd.crosstab(sub[gender_col], sub[approve_col])
print('\nContingency table (gender x approve):')
print(ct)

# Chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-square test: chi2={chi2:.4f}, dof={dof}, p={p:.6f}")

# Difference in proportions and CI
p_f = counts.loc[1, 'mean']
p_m = counts.loc[0, 'mean']

n_f = counts.loc[1, 'count']
n_m = counts.loc[0, 'count']

diff = p_f - p_m
# standard error for difference in proportions
se = np.sqrt(p_f*(1-p_f)/n_f + p_m*(1-p_m)/n_m)
ci_low = diff - 1.96*se
ci_high = diff + 1.96*se
print(f"\nApproval rate female: {p_f:.4f} (n={n_f})")
print(f"Approval rate male:   {p_m:.4f} (n={n_m})")
print(f"Difference (female - male): {diff:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]")

# Logistic regression: approve ~ female
X = sm.add_constant(sub[gender_col])
model = sm.Logit(sub[approve_col], X).fit(disp=False)
print('\nLogit approve ~ female:')
print(model.summary())

# Odds ratio and CI
params = model.params
conf = model.conf_int()

odds_ratio = np.exp(params[gender_col])
conf_or = np.exp(conf.loc[gender_col])
print(f"\nOdds ratio (female vs male): {odds_ratio:.4f}")
print(f"95% CI for OR: [{conf_or[0]:.4f}, {conf_or[1]:.4f}]")
