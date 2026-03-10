import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path


df = pd.read_csv('crofoot.csv')

# relative group size and relative location advantage
# Positive rel_size => focal larger than other
# Positive rel_loc => other farther from its home range center than focal (focal closer to its own center)
df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_loc'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for effect size comparison
for col in ['rel_size', 'rel_loc']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Fit logistic regression with both predictors
model = smf.glm('win ~ rel_size_z + rel_loc_z', data=df, family=sm.families.Binomial()).fit()

# Also fit separate models for each predictor
model_size = smf.glm('win ~ rel_size_z', data=df, family=sm.families.Binomial()).fit()
model_loc = smf.glm('win ~ rel_loc_z', data=df, family=sm.families.Binomial()).fit()

# Compute odds ratios and 95% CI

def odds_ci(m):
    params = m.params
    conf = m.conf_int()
    or_vals = np.exp(params)
    or_ci = np.exp(conf)
    out = pd.DataFrame({
        'coef': params,
        'p': m.pvalues,
        'odds_ratio': or_vals,
        'ci_low': or_ci[0],
        'ci_high': or_ci[1],
    })
    return out

summary = {
    'n': len(df),
    'win_rate': df['win'].mean(),
    'rel_size_mean': df['rel_size'].mean(),
    'rel_loc_mean': df['rel_loc'].mean(),
}

print('SUMMARY', summary)
print('\nMODEL_BOTH')
print(odds_ci(model))
print('\nMODEL_SIZE')
print(odds_ci(model_size))
print('\nMODEL_LOC')
print(odds_ci(model_loc))
