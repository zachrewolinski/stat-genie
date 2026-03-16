import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('crofoot.csv')

# Derived variables
# Relative group size (focal - other)
df['rel_size'] = df['n_focal'] - df['n_other']
# Relative location: positive means focal closer to its home-range center than other is to its own
# (i.e., focal has location advantage)
df['rel_location'] = df['dist_other'] - df['dist_focal']

# Basic sanity
print('Rows:', len(df))
print('Win rate:', df['win'].mean())

# Logistic regression with rel_size and rel_location
X = df[['rel_size', 'rel_location']]
X = sm.add_constant(X)
model = sm.GLM(df['win'], X, family=sm.families.Binomial()).fit()
print('\nGLM (Binomial) win ~ rel_size + rel_location')
print(model.summary())

# Marginal models for each predictor
for col in ['rel_size', 'rel_location']:
    X1 = sm.add_constant(df[[col]])
    m1 = sm.GLM(df['win'], X1, family=sm.families.Binomial()).fit()
    print(f"\nGLM (Binomial) win ~ {col}")
    print(m1.summary())

# Effect sizes: odds ratios
params = model.params
conf = model.conf_int()
print('\nOdds ratios (multivariable):')
for col in ['rel_size', 'rel_location']:
    or_val = params[col].round(4)
    # odds ratio is exp(beta)
    print(col, 'beta', params[col], 'OR', float(np.exp(params[col])))

# Descriptive: mean rel_size and rel_location by win
print('\nMeans by win:')
print(df.groupby('win')[['rel_size','rel_location']].mean())
