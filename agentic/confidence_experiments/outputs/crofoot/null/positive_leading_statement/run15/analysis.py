import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('crofoot.csv')

# Create relative measures

df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_size_ratio'] = df['n_focal'] / df['n_other']
df['rel_location'] = df['dist_other'] - df['dist_focal']
# positive rel_location => contest closer to focal (other further from its center than focal)

# Basic summaries
print('Rows:', len(df))
print(df[['win', 'rel_size', 'rel_location']].describe())

# Logistic regression with difference metrics
model1 = smf.logit('win ~ rel_size + rel_location', data=df).fit(disp=False)
print('\nModel1 (rel_size + rel_location):')
print(model1.summary())

# Logistic with ratio (log)
df['log_size_ratio'] = np.log(df['rel_size_ratio'])
model2 = smf.logit('win ~ log_size_ratio + rel_location', data=df).fit(disp=False)
print('\nModel2 (log size ratio + rel_location):')
print(model2.summary())

# Logistic with dist_focal and dist_other individually
model3 = smf.logit('win ~ rel_size + dist_focal + dist_other', data=df).fit(disp=False)
print('\nModel3 (rel_size + dist_focal + dist_other):')
print(model3.summary())

# Odds ratios for model1
params = model1.params
conf = model1.conf_int()
or_df = pd.DataFrame({
    'coef': params,
    'OR': np.exp(params),
    'OR_low': np.exp(conf[0]),
    'OR_high': np.exp(conf[1]),
    'p': model1.pvalues
})
print('\nModel1 odds ratios:')
print(or_df)

# Standardize predictors for effect comparison
df['rel_size_z'] = (df['rel_size'] - df['rel_size'].mean()) / df['rel_size'].std(ddof=0)
df['rel_location_z'] = (df['rel_location'] - df['rel_location'].mean()) / df['rel_location'].std(ddof=0)
model4 = smf.logit('win ~ rel_size_z + rel_location_z', data=df).fit(disp=False)
print('\nModel4 (standardized):')
print(model4.summary())
