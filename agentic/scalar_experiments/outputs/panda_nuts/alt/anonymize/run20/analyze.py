import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns
# feature1: ID
# feature2: age
# feature3: sex
# feature4: hammer type
# feature5: nuts opened
# feature6: duration (seconds)
# feature7: help

# Basic cleaning
# Standardize categorical values
for col in ['feature3', 'feature4', 'feature7']:
    df[col] = df[col].astype(str).str.strip()

# Create efficiency: nuts opened per minute and per second
# Avoid division by zero

# duration seconds

# Add efficiency

# Only consider rows with positive duration

# Efficiency per second

df = df[df['feature6'] > 0].copy()

df['eff_per_sec'] = df['feature5'] / df['feature6']

df['eff_per_min'] = df['feature5'] / (df['feature6'] / 60.0)

# Encode categorical
# sex: f/m
# help: y/N (note inconsistent case?)

df['sex'] = df['feature3']
df['help'] = df['feature7']

df['sex'] = df['sex'].str.lower()

df['help'] = df['help'].str.lower()

# make sure values are only m/f and y/n

# print unique
print('unique sex:', sorted(df['sex'].unique()))
print('unique help:', sorted(df['help'].unique()))

# Summary
print('n rows:', len(df))
print(df[['feature2','feature5','feature6','eff_per_sec','eff_per_min']].describe())

# Linear regression on efficiency

# Use OLS with robust standard errors

ols = smf.ols('eff_per_sec ~ feature2 + C(sex) + C(help)', data=df).fit(cov_type='HC3')
print('\nOLS eff_per_sec:')
print(ols.summary())

# GLM Poisson for counts with offset log(duration)
# model nuts opened as count

# For offset, duration in seconds; use log
# Add small epsilon if any zero duration (filtered)

glm_pois = smf.glm('feature5 ~ feature2 + C(sex) + C(help)', data=df,
                   family=sm.families.Poisson(),
                   offset=np.log(df['feature6'])).fit(cov_type='HC3')
print('\nPoisson GLM count with offset log(duration):')
print(glm_pois.summary())

# Check overdispersion: Pearson chi2 / df

pearson_chi2 = glm_pois.pearson_chi2
ratio = pearson_chi2 / glm_pois.df_resid
print('\nPoisson overdispersion ratio:', ratio)

# Negative binomial GLM if overdispersion

# statsmodels GLM NB uses alpha param; we can estimate with NB2 (discrete) maybe.

# Use statsmodels NegativeBinomial (discrete) with log(duration) offset.

from statsmodels.discrete.discrete_model import NegativeBinomial

# Design matrix manually

# Use pandas get_dummies for categorical

X = pd.get_dummies(df[['feature2','sex','help']], drop_first=True)
# Add constant
X = sm.add_constant(X)

nb = NegativeBinomial(df['feature5'], X, offset=np.log(df['feature6']))
nb_res = nb.fit(disp=0)
print('\nNegative Binomial count with offset log(duration):')
print(nb_res.summary())

# Store key coefficients and p-values

print('\nCoefficients (NB):')
print(nb_res.params)
print('\nP-values (NB):')
print(nb_res.pvalues)

# Also test for interactions maybe? not necessary.
