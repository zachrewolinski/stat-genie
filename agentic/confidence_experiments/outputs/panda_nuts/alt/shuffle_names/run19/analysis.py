import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns to semantic names
# age column is chimpanzee ID (not used for analysis)
# hammer column is actual age (years)
# nuts_opened column is sex (m/f)
# sex column is hammer type (not used)
# help column is number of nuts opened
# chimpanzee column is duration in seconds
# seconds column is help indicator (y/N)

# Build analysis dataframe
analysis = pd.DataFrame({
    'age': df['hammer'],
    'sex': df['nuts_opened'],
    'helped': df['seconds'],
    'nuts_opened': df['help'],
    'seconds': df['chimpanzee'],
})

analysis['efficiency'] = analysis['nuts_opened'] / analysis['seconds']

# Basic summaries
summary = {
    'n': len(analysis),
    'efficiency_mean': analysis['efficiency'].mean(),
    'efficiency_std': analysis['efficiency'].std(),
    'efficiency_min': analysis['efficiency'].min(),
    'efficiency_max': analysis['efficiency'].max(),
}
print('SUMMARY', summary)

# Group means
print('SEX MEANS')
print(analysis.groupby('sex')['efficiency'].agg(['mean','std','count']))
print('HELP MEANS')
print(analysis.groupby('helped')['efficiency'].agg(['mean','std','count']))

# OLS regression with categorical sex and help
model = smf.ols('efficiency ~ age + C(sex) + C(helped)', data=analysis).fit()
print(model.summary())

# Also standardized coefficients (for effect size)
# Standardize numeric predictors and outcome
analysis_std = analysis.copy()
analysis_std['age_z'] = (analysis_std['age'] - analysis_std['age'].mean()) / analysis_std['age'].std()
analysis_std['efficiency_z'] = (analysis_std['efficiency'] - analysis_std['efficiency'].mean()) / analysis_std['efficiency'].std()
model_std = smf.ols('efficiency_z ~ age_z + C(sex) + C(helped)', data=analysis_std).fit()
print('STD COEFS')
print(model_std.params)
print(model_std.pvalues)

