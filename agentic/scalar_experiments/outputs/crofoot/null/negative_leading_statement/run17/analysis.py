import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('crofoot.csv')

# Create predictors
# Relative group size (focal - other)
df['rel_size'] = df['n_focal'] - df['n_other']
# Location advantage: positive means contest closer to focal home range center
# because other is farther from its own center

df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Simple descriptives
print('N:', len(df))
print(df[['win','rel_size','loc_adv']].describe())

# Logistic regression: win ~ rel_size + loc_adv
X = df[['rel_size','loc_adv']]
X = sm.add_constant(X)
model = sm.Logit(df['win'], X)
try:
    res = model.fit(disp=False)
except Exception as e:
    print('Logit failed:', e)
    res = model.fit_regularized(disp=False)

print(res.summary())

# Compute odds ratios and 95% CI
params = res.params
conf = res.conf_int()
or_vals = np.exp(params)
or_conf = np.exp(conf)
print('\nOdds Ratios:')
print(pd.DataFrame({'OR': or_vals, 'CI_low': or_conf[0], 'CI_high': or_conf[1], 'p': res.pvalues}))

# Also test each predictor individually
for col in ['rel_size','loc_adv']:
    X1 = sm.add_constant(df[[col]])
    m1 = sm.Logit(df['win'], X1)
    try:
        r1 = m1.fit(disp=False)
    except Exception as e:
        r1 = m1.fit_regularized(disp=False)
    print(f"\nUnivariate logit for {col}")
    print(r1.summary())
    or1 = np.exp(r1.params)
    ci1 = np.exp(r1.conf_int())
    print(pd.DataFrame({'OR': or1, 'CI_low': ci1[0], 'CI_high': ci1[1], 'p': r1.pvalues}))

# Nonparametric checks: compare rel_size and loc_adv between wins/losses
wins = df[df['win']==1]
losses = df[df['win']==0]

for col in ['rel_size','loc_adv']:
    t_stat, t_p = stats.ttest_ind(wins[col], losses[col], equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(wins[col], losses[col], alternative='two-sided')
    print(f"\n{col} mean win={wins[col].mean():.3f} loss={losses[col].mean():.3f}")
    print(f"t-test p={t_p:.4f}, Mann-Whitney p={u_p:.4f}")

# Effect size (Cohen's d)
for col in ['rel_size','loc_adv']:
    mean_diff = wins[col].mean() - losses[col].mean()
    # pooled SD
    sd_pooled = np.sqrt(((wins[col].var(ddof=1) + losses[col].var(ddof=1))/2))
    d = mean_diff / sd_pooled if sd_pooled != 0 else np.nan
    print(f"Cohen's d for {col}: {d:.3f}")
