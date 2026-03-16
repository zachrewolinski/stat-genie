import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric types
for col in ['masfem','masfem_mturk','gender_mf','alldeaths','category','min','wind','ndam','ndam15','year']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create log deaths to handle skew
# Add 1 to include zero deaths
if 'alldeaths' in df.columns:
    df['log_deaths'] = np.log1p(df['alldeaths'])

# Examine correlations
corrs = df[['masfem','masfem_mturk','gender_mf','alldeaths','log_deaths','category','min','wind','ndam15','year']].corr()

# Regression models
results = {}

# Model 1: unadjusted log deaths ~ masfem
m1 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')
results['m1'] = m1

# Model 2: log deaths ~ masfem + intensity controls
# Use category, min pressure, wind, ndam15, year
# Drop rows with missing in these columns
cols = ['log_deaths','masfem','category','min','wind','ndam15','year']
df2 = df[cols].dropna()
m2 = smf.ols('log_deaths ~ masfem + category + min + wind + ndam15 + year', data=df2).fit(cov_type='HC3')
results['m2'] = m2

# Model 3: using gender_mf instead of masfem
cols3 = ['log_deaths','gender_mf','category','min','wind','ndam15','year']
df3 = df[cols3].dropna()
m3 = smf.ols('log_deaths ~ gender_mf + category + min + wind + ndam15 + year', data=df3).fit(cov_type='HC3')
results['m3'] = m3

# Also consider interaction (masfem * category) as in some prior claims
m4 = smf.ols('log_deaths ~ masfem * category + min + wind + ndam15 + year', data=df2).fit(cov_type='HC3')
results['m4'] = m4

# Summaries for key coefficients
summary = {
    'n_rows': len(df),
    'corr_masfem_log_deaths': corrs.loc['masfem','log_deaths'],
    'corr_gender_log_deaths': corrs.loc['gender_mf','log_deaths'],
}

# Extract coefficient and p-value for masfem in models
summary['m1_masfem_coef'] = m1.params.get('masfem', np.nan)
summary['m1_masfem_p'] = m1.pvalues.get('masfem', np.nan)
summary['m2_masfem_coef'] = m2.params.get('masfem', np.nan)
summary['m2_masfem_p'] = m2.pvalues.get('masfem', np.nan)
summary['m3_gender_coef'] = m3.params.get('gender_mf', np.nan)
summary['m3_gender_p'] = m3.pvalues.get('gender_mf', np.nan)
summary['m4_masfem_coef'] = m4.params.get('masfem', np.nan)
summary['m4_masfem_p'] = m4.pvalues.get('masfem', np.nan)
summary['m4_interaction_coef'] = m4.params.get('masfem:category', np.nan)
summary['m4_interaction_p'] = m4.pvalues.get('masfem:category', np.nan)
summary['n_m2'] = int(m2.nobs)
summary['n_m3'] = int(m3.nobs)

print(summary)

# Save detailed outputs for inspection
with open('analysis_output.txt', 'w') as f:
    f.write('Correlation matrix (selected):\n')
    f.write(corrs.to_string())
    f.write('\n\nModel 1: log_deaths ~ masfem (HC3)\n')
    f.write(m1.summary().as_text())
    f.write('\n\nModel 2: log_deaths ~ masfem + controls (HC3)\n')
    f.write(m2.summary().as_text())
    f.write('\n\nModel 3: log_deaths ~ gender_mf + controls (HC3)\n')
    f.write(m3.summary().as_text())
    f.write('\n\nModel 4: log_deaths ~ masfem*category + controls (HC3)\n')
    f.write(m4.summary().as_text())

