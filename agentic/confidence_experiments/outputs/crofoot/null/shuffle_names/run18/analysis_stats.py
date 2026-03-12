import pandas as pd
import numpy as np
import statsmodels.api as sm

# load data

df = pd.read_csv('crofoot.csv')

# variable mapping (focal win outcome, sizes, distances)
# m_focal is binary outcome
outcome = df['m_focal']

# choose group size columns (5-13 range) and distance columns (55-853 range)
size_focal = df['f_other']
size_other = df['win']
loc_focal = df['m_other']
loc_other = df['n_focal']

size_diff = size_focal - size_other
loc_adv = loc_other - loc_focal

# standardize predictors
size_z = (size_diff - size_diff.mean()) / size_diff.std(ddof=0)
loc_z = (loc_adv - loc_adv.mean()) / loc_adv.std(ddof=0)

X = sm.add_constant(pd.DataFrame({'size_diff_z': size_z, 'loc_adv_z': loc_z}))
model = sm.Logit(outcome, X).fit(disp=False)

# extract coefficients, p-values, odds ratios
coef = model.params
pvals = model.pvalues
conf = model.conf_int()

# odds ratios for 1 SD change
or_vals = np.exp(coef)
or_ci = np.exp(conf)

print('coefficients')
print(coef)
print('\nP-values')
print(pvals)
print('\nOdds ratios (1 SD)')
print(or_vals)
print('\nOR 95% CI')
print(or_ci)

# descriptive win rates
size_pos = outcome[size_diff > 0].mean()
size_neg = outcome[size_diff <= 0].mean()
loc_pos = outcome[loc_adv > 0].mean()
loc_neg = outcome[loc_adv <= 0].mean()

print('\nWin rate focal larger (size_diff>0):', size_pos)
print('Win rate focal not larger (<=0):', size_neg)
print('Win rate focal closer (loc_adv>0):', loc_pos)
print('Win rate focal not closer (<=0):', loc_neg)

# correlation (point-biserial / Pearson)
print('\nCorrelation outcome with size_diff and loc_adv (Pearson):')
print('size_diff', np.corrcoef(outcome, size_diff)[0,1])
print('loc_adv', np.corrcoef(outcome, loc_adv)[0,1])
