import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import json

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Compute efficiency: nuts opened per second
# Avoid division by zero just in case
# seconds can be float

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Basic info
print('rows', len(df))
print('chimpanzee unique', df['chimpanzee'].nunique())
print('sessions per chimp (min/max)', df.groupby('chimpanzee').size().min(), df.groupby('chimpanzee').size().max())
print('efficiency summary', df['efficiency'].describe())

# Encode categorical for regression; statsmodels handles categorical via C()

# Mixed effects model with random intercept for chimpanzee
# Using formula; treat sex and help as categorical
# Age numeric

# Use MixedLM
try:
    md = smf.mixedlm('efficiency ~ age + C(sex) + C(help)', data=df, groups=df['chimpanzee'])
    mdf = md.fit(reml=False)
    print('\nMixedLM results')
    print(mdf.summary())
except Exception as e:
    print('MixedLM failed:', e)
    mdf = None

# Also OLS as comparison
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()
print('\nOLS results')
print(ols.summary())

# Provide robust SEs for OLS maybe
ols_robust = ols.get_robustcov_results(cov_type='HC3')
print('\nOLS robust HC3 results')
print(ols_robust.summary())

# Check group means for help/sex
print('\nMean efficiency by sex')
print(df.groupby('sex')['efficiency'].mean())
print('\nMean efficiency by help')
print(df.groupby('help')['efficiency'].mean())

# Correlation with age
print('\nCorrelation age-efficiency', df['age'].corr(df['efficiency']))

