import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import logit

# Load data
_df = pd.read_csv('crofoot.csv')

# Keep relevant columns
# compute relative size and relative location
df = _df.copy()

# check columns exist
required = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# define predictors
df['size_diff'] = df['n_focal'] - df['n_other']
# location advantage: positive if other farther from its center than focal (contest closer to focal)
df['dist_diff'] = df['dist_other'] - df['dist_focal']

# Basic summary
print('N rows:', len(df))
print('Win mean:', df['win'].mean())
print('Size diff summary:', df['size_diff'].describe())
print('Dist diff summary:', df['dist_diff'].describe())

# logistic regression
model = logit('win ~ size_diff + dist_diff', data=df).fit(disp=False)
print(model.summary())

# compute odds ratios and CI
params = model.params
conf = model.conf_int()
conf.columns = ['2.5%', '97.5%']

odds = np.exp(params)
conf_odds = np.exp(conf)

print('\nOdds ratios:')
print(odds)
print('\n95% CI for odds ratios:')
print(conf_odds)

# Wald p-values
print('\nP-values:')
print(model.pvalues)

# Likelihood ratio test vs null
null_model = logit('win ~ 1', data=df).fit(disp=False)
llf_full = model.llf
llf_null = null_model.llf
lr_stat = 2*(llf_full - llf_null)
from scipy.stats import chi2
p_lr = chi2.sf(lr_stat, df=2)
print('\nLR test stat:', lr_stat, 'p:', p_lr)

# Also test each predictor individually
model_size = logit('win ~ size_diff', data=df).fit(disp=False)
model_dist = logit('win ~ dist_diff', data=df).fit(disp=False)
print('\nModel size only:')
print(model_size.summary())
print('\nModel dist only:')
print(model_dist.summary())
