import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map actual variables using info.json descriptions (shuffled column names)
# age in years -> column 'hammer'
# sex -> column 'nuts_opened' (values f/m)
# help received -> column 'seconds' (y/N)
# nuts opened -> column 'help'
# session duration seconds -> column 'chimpanzee'

analysis = pd.DataFrame({
    'age_years': df['hammer'].astype(float),
    'sex': df['nuts_opened'].astype(str),
    'help_received': df['seconds'].astype(str),
    'nuts_opened': df['help'].astype(float),
    'duration_sec': df['chimpanzee'].astype(float),
})

# Efficiency: nuts opened per second
analysis['efficiency'] = analysis['nuts_opened'] / analysis['duration_sec']

# Basic sanity
summary = {
    'n': len(analysis),
    'sex_counts': analysis['sex'].value_counts().to_dict(),
    'help_counts': analysis['help_received'].value_counts().to_dict(),
    'efficiency_mean': analysis['efficiency'].mean(),
    'efficiency_median': analysis['efficiency'].median(),
}

print('SUMMARY', summary)

# OLS with robust SE
model = smf.ols('efficiency ~ age_years + C(sex) + C(help_received)', data=analysis).fit(cov_type='HC3')
print(model.summary())

# Collect key stats
params = model.params
pvals = model.pvalues

print('\nCOEFFICIENTS')
for k in params.index:
    print(k, params[k], pvals[k])

# Simple group means for help and sex
print('\nGROUP MEANS')
print(analysis.groupby('sex')['efficiency'].mean())
print(analysis.groupby('help_received')['efficiency'].mean())

