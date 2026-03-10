import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data with header row
csv_path = 'crofoot.csv'

df = pd.read_csv(csv_path)

# Ensure numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with missing values in key columns
key_cols = ['feature4', 'feature5', 'feature6', 'feature7', 'feature8']
sub = df.dropna(subset=key_cols).copy()

# Define variables
sub['rel_group_size'] = sub['feature7'] - sub['feature8']
# Relative location: positive means contest farther from focal center (closer to other), negative means closer to focal
sub['rel_location'] = sub['feature5'] - sub['feature6']

# Outcome
y = sub['feature4']

# Logistic regression with both predictors
X = sub[['rel_group_size', 'rel_location']]
X = sm.add_constant(X)
model = sm.Logit(y, X).fit(disp=False)

# Univariate models
X1 = sm.add_constant(sub[['rel_group_size']])
model_size = sm.Logit(y, X1).fit(disp=False)
X2 = sm.add_constant(sub[['rel_location']])
model_loc = sm.Logit(y, X2).fit(disp=False)

print('N:', len(sub))
print('\nDescriptive:')
print(sub[['feature4','rel_group_size','rel_location']].describe())

print('\nLogit (both predictors):')
print(model.summary())

print('\nLogit (rel_group_size only):')
print(model_size.summary())

print('\nLogit (rel_location only):')
print(model_loc.summary())

# Odds ratios and CI
params = model.params
conf = model.conf_int()
conf.columns = ['2.5%', '97.5%']
OR = np.exp(params)
OR_ci = np.exp(conf)

print('\nOdds Ratios (both predictors):')
print(pd.DataFrame({'OR': OR, '2.5%': OR_ci['2.5%'], '97.5%': OR_ci['97.5%'], 'p': model.pvalues}))

# Simple effect sizes: correlation between predictors and outcome
corr_size = np.corrcoef(sub['rel_group_size'], y)[0,1]
corr_loc = np.corrcoef(sub['rel_location'], y)[0,1]
print('\nPoint-biserial correlations:')
print({'rel_group_size': corr_size, 'rel_location': corr_loc})
