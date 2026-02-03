import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Basic prep
# Use log1p for deaths to reduce skew, common in hurricane analyses
# Keep relevant predictors

# Some columns might have missing or non-numeric values; coerce where needed
for col in ['masfem', 'wind', 'min', 'category', 'alldeaths', 'ndam15', 'ndam']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing required values
req = ['masfem', 'alldeaths', 'wind', 'min', 'category']
work = df.dropna(subset=req).copy()
work['log_deaths'] = np.log1p(work['alldeaths'])

# Model 1: bivariate association
m1 = smf.ols('log_deaths ~ masfem', data=work).fit()

# Model 2: control for storm intensity (wind, pressure, category)
m2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=work).fit()

# Model 3: add damage as proxy for exposure/impact (ndam15) if available
if 'ndam15' in work.columns:
    m3 = smf.ols('log_deaths ~ masfem + wind + min + category + ndam15', data=work).fit()
else:
    m3 = None

# Print key results
print('N:', len(work))
print('\nModel 1: log_deaths ~ masfem')
print(m1.summary().tables[1])

print('\nModel 2: + wind + min + category')
print(m2.summary().tables[1])

if m3 is not None:
    print('\nModel 3: + ndam15')
    print(m3.summary().tables[1])

# Also compute correlation between masfem and deaths for context
corr = work['masfem'].corr(work['alldeaths'])
print('\nCorrelation (masfem, alldeaths):', corr)
