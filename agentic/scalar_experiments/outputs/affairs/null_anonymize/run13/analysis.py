import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# Map columns
affairs = df['feature2'].astype(float)
children = df['feature6'].astype(str).str.strip().str.lower()

# Split
mask_yes = children == 'yes'
mask_no = children == 'no'

if mask_yes.sum() == 0 or mask_no.sum() == 0:
    raise ValueError('Missing yes/no groups for children')

# Means
mean_yes = affairs[mask_yes].mean()
mean_no = affairs[mask_no].mean()
std_yes = affairs[mask_yes].std(ddof=1)
std_no = affairs[mask_no].std(ddof=1)

n_yes = mask_yes.sum()
n_no = mask_no.sum()

# Welch t-test
res = stats.ttest_ind(affairs[mask_no], affairs[mask_yes], equal_var=False)

# Cohen's d (pooled sd)
pooled_sd = np.sqrt(((n_no - 1) * std_no ** 2 + (n_yes - 1) * std_yes ** 2) / (n_no + n_yes - 2))
cohens_d = (mean_no - mean_yes) / pooled_sd

# Any-affair proportions
any_affair = (affairs > 0).astype(int)
prop_no = any_affair[mask_no].mean()
prop_yes = any_affair[mask_yes].mean()

# Two-proportion z-test
p_pool = (any_affair[mask_no].sum() + any_affair[mask_yes].sum()) / (n_no + n_yes)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_no + 1 / n_yes))
if se_pool == 0:
    z = 0.0
    p_prop = 1.0
else:
    z = (prop_no - prop_yes) / se_pool
    p_prop = 2 * (1 - stats.norm.cdf(abs(z)))

# Cohen's h for proportions
cohens_h = 2 * np.arcsin(np.sqrt(prop_no)) - 2 * np.arcsin(np.sqrt(prop_yes))

# Direction: positive means fewer affairs with children
# If mean_no > mean_yes and prop_no > prop_yes => positive
# Use sign from average of mean and proportion difference
mean_diff = mean_no - mean_yes
prop_diff = prop_no - prop_yes

direction = np.sign(mean_diff + prop_diff)
if direction == 0:
    direction = 1.0

# Effect strength combining d and h (capped)
strength_d = min(abs(cohens_d) / 1.5, 1.0)  # d=1.5 maps to 1
strength_h = min(abs(cohens_h) / 1.0, 1.0)  # h=1 maps to 1
strength = 0.5 * (strength_d + strength_h)

# Significance multiplier
# Use worse (larger) p-value to be conservative
p_worst = max(res.pvalue, p_prop)
if p_worst < 0.01:
    sig_mult = 1.0
elif p_worst < 0.05:
    sig_mult = 0.85
elif p_worst < 0.1:
    sig_mult = 0.7
else:
    sig_mult = 0.4

score = int(round(direction * 100 * strength * sig_mult))
score = int(max(-100, min(100, score)))

print('n_yes', n_yes, 'n_no', n_no)
print('mean_yes', mean_yes, 'mean_no', mean_no, 'diff(no-yes)', mean_diff)
print('std_yes', std_yes, 'std_no', std_no)
print('t_stat', res.statistic, 'p_value', res.pvalue)
print('cohens_d', cohens_d)
print('prop_yes', prop_yes, 'prop_no', prop_no, 'diff(no-yes)', prop_diff)
print('z', z, 'p_prop', p_prop)
print('cohens_h', cohens_h)
print('direction', direction, 'strength', strength, 'sig_mult', sig_mult)
print('score', score)
