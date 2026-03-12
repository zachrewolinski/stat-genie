import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic cleaning
# Drop rows with missing in key variables
key_cols = ['masfem','alldeaths','wind','min','category','ndam15','gender_mf']

summary = {}
summary['n_rows'] = len(df)
summary['missing'] = df[key_cols].isna().sum().to_dict()

# Outcome transformations
# Use log1p deaths to handle zeros and skewness

df['log_deaths'] = np.log1p(df['alldeaths'])

# Severity index: standardize wind (higher worse), min pressure (lower worse) -> invert min
# We'll create a simple severity z-score by combining wind and -min

df['min_inverted'] = -df['min']

for col in ['wind','min_inverted']:
    df[col+'_z'] = (df[col] - df[col].mean())/df[col].std(ddof=0)

df['severity_z'] = df[['wind_z','min_inverted_z']].mean(axis=1)

# Models
results = {}

# Model 1: simple correlation and OLS with masfem only
model1 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')
results['model1'] = model1

# Model 2: control for category, wind, min pressure
model2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit(cov_type='HC3')
results['model2'] = model2

# Model 3: control for severity_z and category
model3 = smf.ols('log_deaths ~ masfem + severity_z + category', data=df).fit(cov_type='HC3')
results['model3'] = model3

# Model 4: include economic damage as proxy for exposure/severity
# Use log1p damage (ndam15)
df['log_ndam15'] = np.log1p(df['ndam15'])
model4 = smf.ols('log_deaths ~ masfem + wind + min + category + log_ndam15', data=df).fit(cov_type='HC3')
results['model4'] = model4

# Poisson regression for counts (with robust SE)
# Add 1e-6 to avoid issues? not needed.
poisson = smf.glm('alldeaths ~ masfem + wind + min + category', data=df,
                  family=sm.families.Poisson()).fit(cov_type='HC3')
results['poisson'] = poisson

# Spearman correlation between masfem and log_deaths
spearman_corr = stats.spearmanr(df['masfem'], df['log_deaths'])
pearson_corr = stats.pearsonr(df['masfem'], df['log_deaths'])

# Partial correlation using residuals: regress log_deaths on controls, masfem on controls, then corr residuals
controls = ['wind','min','category']
Xc = sm.add_constant(df[controls])
res_y = sm.OLS(df['log_deaths'], Xc).fit().resid
res_x = sm.OLS(df['masfem'], Xc).fit().resid
partial_pearson = stats.pearsonr(res_x, res_y)

# Collect key outputs
out = {
    'summary': summary,
    'spearman_corr': spearman_corr,
    'pearson_corr': pearson_corr,
    'partial_pearson_controls_wind_min_cat': partial_pearson,
}

# Extract coefficient table for masfem in each model
coef_table = {}
for name, m in results.items():
    if 'masfem' in m.params.index:
        coef_table[name] = {
            'coef': float(m.params['masfem']),
            'se': float(m.bse['masfem']),
            'pvalue': float(m.pvalues['masfem']),
        }

out['masfem_effects'] = coef_table

# Save outputs for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

# Also save model summaries text
with open('analysis_summaries.txt','w') as f:
    for name, m in results.items():
        f.write(f"\n{name}\n")
        f.write(m.summary().as_text())
        f.write("\n")

print('done')
