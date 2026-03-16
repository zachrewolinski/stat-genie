import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

df = pd.read_csv('hurricane.csv')
print(df.head())
print(df.describe(include='all'))

# Rename columns for convenience
cols = {
    'feature1':'id',
    'feature2':'year',
    'feature3':'name',
    'feature4':'fem_index',
    'feature5':'min_pressure',
    'feature6':'female_binary',
    'feature7':'category',
    'feature8':'deaths',
    'feature9':'damage_2013',
    'feature10':'years_since',
    'feature11':'source',
    'feature12':'fem_index_mturk',
    'feature13':'max_wind',
    'feature14':'damage_2015',
}

df = df.rename(columns=cols)

# Basic correlation
print('corr fem_index vs deaths:', df['fem_index'].corr(df['deaths']))
print('corr fem_index_mturk vs deaths:', df['fem_index_mturk'].corr(df['deaths']))

# log transform for deaths and damage
for col in ['deaths','damage_2013','damage_2015']:
    df[f'log1p_{col}'] = np.log1p(df[col])

# Simple OLS: log deaths on fem_index
X = sm.add_constant(df['fem_index'])
model1 = sm.OLS(df['log1p_deaths'], X).fit(cov_type='HC3')
print('Model1 log deaths ~ fem_index')
print(model1.summary())

# Control for intensity: min_pressure, category, max_wind
X2 = df[['fem_index','min_pressure','category','max_wind']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(df['log1p_deaths'], X2).fit(cov_type='HC3')
print('Model2 log deaths ~ fem_index + intensity')
print(model2.summary())

# Using binary female
X3 = df[['female_binary','min_pressure','category','max_wind']]
X3 = sm.add_constant(X3)
model3 = sm.OLS(df['log1p_deaths'], X3).fit(cov_type='HC3')
print('Model3 log deaths ~ female_binary + intensity')
print(model3.summary())

# Interaction: fem_index * intensity (category)
df['fem_x_cat'] = df['fem_index'] * df['category']
X4 = df[['fem_index','category','fem_x_cat','min_pressure','max_wind']]
X4 = sm.add_constant(X4)
model4 = sm.OLS(df['log1p_deaths'], X4).fit(cov_type='HC3')
print('Model4 log deaths ~ fem_index + category + interaction + intensity')
print(model4.summary())

# Additional model using fem_index_mturk
X5 = df[['fem_index_mturk','min_pressure','category','max_wind']]
X5 = sm.add_constant(X5)
model5 = sm.OLS(df['log1p_deaths'], X5).fit(cov_type='HC3')
print('Model5 log deaths ~ fem_index_mturk + intensity')
print(model5.summary())

# optional: negative binomial on deaths (count), using statsmodels discrete
import statsmodels.discrete.discrete_model as smd

endog = df['deaths']
exog = sm.add_constant(df[['fem_index','min_pressure','category','max_wind']])
try:
    nb = smd.NegativeBinomial(endog, exog).fit(disp=False)
    print('NegativeBinomial deaths ~ fem_index + intensity')
    print(nb.summary())
except Exception as e:
    print('NB failed', e)

# Save key results to a csv for easier parsing
results = []
for name, model in [('model1', model1), ('model2', model2), ('model3', model3), ('model4', model4), ('model5', model5)]:
    for param in model.params.index:
        results.append({
            'model': name,
            'param': param,
            'coef': model.params[param],
            'pvalue': model.pvalues[param],
            'se': model.bse[param]
        })

pd.DataFrame(results).to_csv('analysis_results.csv', index=False)
