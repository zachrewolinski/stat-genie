import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Map columns based on observed values
_df = _df.rename(columns={
    'sockets': 'tooth_class',      # Anterior/Posterior/Premolar
    'prob_male': 'specimen_id',    # specimen identifier
    'tooth_class': 'genus_cat',    # Homo sapiens / Pan / Papio / Pongo
    'pop': 'age_at_death',         # continuous 8-71
    'stdev_age': 'prob_male',      # 0-1 in increments
})

# Response (AMTL measure) is in 'genus' column (continuous, varies by tooth class)
_df['amtl_measure'] = _df['genus']

# Create human indicator
_df['human'] = (_df['genus_cat'] == 'Homo sapiens').astype(int)

# Basic summaries
summary = _df.groupby('genus_cat')['amtl_measure'].agg(['mean','std','count'])
print('AMTL measure by genus (raw):')
print(summary)

# Model 1: OLS on amtl_measure with clustered SE by specimen
formula = 'amtl_measure ~ human + age_at_death + prob_male + C(tooth_class)'
model1 = smf.ols(formula, data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['specimen_id']})
print('\nModel 1 (OLS on amtl_measure)')
print(model1.summary().tables[1])

# Model 2: transform amtl_measure via logistic to approximate frequency (if amtl_measure is logit)
_df['amtl_freq'] = 1/(1+np.exp(-_df['amtl_measure']))
model2 = smf.ols('amtl_freq ~ human + age_at_death + prob_male + C(tooth_class)', data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['specimen_id']})
print('\nModel 2 (OLS on logistic(amtl_measure))')
print(model2.summary().tables[1])

# Extract human effect stats
human_coef1 = model1.params['human']
human_p1 = model1.pvalues['human']

human_coef2 = model2.params['human']
human_p2 = model2.pvalues['human']

print('\nHuman effect (model1): coef', human_coef1, 'p', human_p1)
print('Human effect (model2): coef', human_coef2, 'p', human_p2)

# Predicted difference in amtl_freq at mean covariates
mean_cov = {
    'human': [0,1],
    'age_at_death': [_df['age_at_death'].mean()]*2,
    'prob_male': [_df['prob_male'].mean()]*2,
    'tooth_class': ['Anterior']*2,
}
mean_df = pd.DataFrame(mean_cov)
# Use model2 for freq difference
pred = model2.predict(mean_df)
print('\nPredicted amtl_freq at means (Anterior): non-human, human', pred.tolist())

