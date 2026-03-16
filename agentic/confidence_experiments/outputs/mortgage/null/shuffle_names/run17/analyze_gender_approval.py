import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('mortgage.csv')

# Gender indicator: column with description "1 if applicant is female" per info.json is 'denied_PMI'
# Approval indicator: column with description "1 if mortgage application was accepted" per info.json is 'deny'

df = _df[['denied_PMI', 'deny']].copy()
# Drop rows with missing

df = df.dropna()

# Rename for clarity

df = df.rename(columns={'denied_PMI': 'female', 'deny': 'approve'})

# contingency table
ct = pd.crosstab(df['female'], df['approve'])

# approval rates
rate_female = ct.loc[1, 1] / ct.loc[1].sum() if 1 in ct.index else np.nan
rate_male = ct.loc[0, 1] / ct.loc[0].sum() if 0 in ct.index else np.nan

# difference in proportions (female - male) and Wald CI
n_f = ct.loc[1].sum() if 1 in ct.index else 0
n_m = ct.loc[0].sum() if 0 in ct.index else 0
p_f = rate_female
p_m = rate_male

# pooled standard error for difference
se_diff = np.sqrt(p_f * (1 - p_f) / n_f + p_m * (1 - p_m) / n_m)
ci_low = (p_f - p_m) - 1.96 * se_diff
ci_high = (p_f - p_m) + 1.96 * se_diff

# chi-square test of independence
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# logistic regression (approve ~ female)
X = sm.add_constant(df['female'])
model = sm.Logit(df['approve'], X, missing='drop')
res = model.fit(disp=False)

# odds ratio
odds_ratio = np.exp(res.params['female'])
ci = res.conf_int().loc['female']
ci_or = np.exp(ci)

print('n', len(df))
print('contingency table:\n', ct)
print('approval rate female', rate_female)
print('approval rate male', rate_male)
print('difference female - male', p_f - p_m)
print('diff 95% CI', (ci_low, ci_high))
print('chi2 p', p_chi)
print('logit coef female', res.params['female'], 'p', res.pvalues['female'])
print('odds ratio', odds_ratio, 'OR 95% CI', (ci_or[0], ci_or[1]))
