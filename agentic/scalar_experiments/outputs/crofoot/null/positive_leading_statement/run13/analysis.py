import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

# Load data
path = Path('crofoot.csv')
df = pd.read_csv(path)

# Create predictors
# Relative group size
# difference and ratio

df['rel_size_diff'] = df['n_focal'] - df['n_other']
df['rel_size_ratio'] = df['n_focal'] / df['n_other']

# Location advantage: contest closer to focal home range center
# smaller dist_focal means closer to focal; so advantage = dist_other - dist_focal

df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for comparability
for col in ['rel_size_diff', 'rel_size_ratio', 'loc_adv']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Basic summaries
summary = {
    'n': len(df),
    'win_rate': df['win'].mean(),
    'rel_size_diff_mean': df['rel_size_diff'].mean(),
    'loc_adv_mean': df['loc_adv'].mean(),
}

# Logistic regression models
# Model 1: size only
m1 = smf.glm('win ~ rel_size_diff_z', data=df, family=sm.families.Binomial()).fit()
# Model 2: location only
m2 = smf.glm('win ~ loc_adv_z', data=df, family=sm.families.Binomial()).fit()
# Model 3: both
m3 = smf.glm('win ~ rel_size_diff_z + loc_adv_z', data=df, family=sm.families.Binomial()).fit()
# Model 4: ratio + loc
m4 = smf.glm('win ~ rel_size_ratio_z + loc_adv_z', data=df, family=sm.families.Binomial()).fit()

# Extract results

def coef_table(model):
    params = model.params
    se = model.bse
    p = model.pvalues
    conf = model.conf_int()
    return pd.DataFrame({
        'coef': params,
        'se': se,
        'p': p,
        'ci_low': conf[0],
        'ci_high': conf[1],
        'odds_ratio': np.exp(params),
        'or_ci_low': np.exp(conf[0]),
        'or_ci_high': np.exp(conf[1]),
    })

results = {
    'summary': summary,
    'm1': coef_table(m1),
    'm2': coef_table(m2),
    'm3': coef_table(m3),
    'm4': coef_table(m4),
    'm1_aic': m1.aic,
    'm2_aic': m2.aic,
    'm3_aic': m3.aic,
    'm4_aic': m4.aic,
}

# Simple correlations (point biserial)
from scipy.stats import pointbiserialr

corr_size = pointbiserialr(df['win'], df['rel_size_diff'])
corr_loc = pointbiserialr(df['win'], df['loc_adv'])
results['corr_size'] = {'r': corr_size.correlation, 'p': corr_size.pvalue}
results['corr_loc'] = {'r': corr_loc.correlation, 'p': corr_loc.pvalue}

# Save results to csvs for inspection
results['m1'].to_csv('m1.csv')
results['m2'].to_csv('m2.csv')
results['m3'].to_csv('m3.csv')
results['m4'].to_csv('m4.csv')

# Print key results
print('SUMMARY', summary)
print('\nM1\n', results['m1'])
print('\nM2\n', results['m2'])
print('\nM3\n', results['m3'])
print('\nM4\n', results['m4'])
print('\nAICs', results['m1_aic'], results['m2_aic'], results['m3_aic'], results['m4_aic'])
print('\nCorr size', results['corr_size'])
print('Corr loc', results['corr_loc'])
