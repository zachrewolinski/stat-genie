import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Rename columns for clarity
cols = {
    'feature4': 'masfem',  # masculinity-femininity index (higher=feminine)
    'feature6': 'female_binary',
    'feature7': 'category',
    'feature5': 'min_pressure',
    'feature13': 'max_wind',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature14': 'damage_2015',
    'feature2': 'year'
}

# keep needed columns
for k, v in cols.items():
    if k in df.columns:
        df = df.rename(columns={k: v})

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = ['masfem', 'female_binary', 'category', 'min_pressure', 'max_wind', 'deaths', 'damage_2013', 'damage_2015', 'year']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows with missing key variables; add damage if present
key_cols = ['deaths', 'masfem', 'category', 'min_pressure', 'max_wind']
if 'damage_2013' in df.columns:
    key_cols.append('damage_2013')
elif 'damage_2015' in df.columns:
    key_cols.append('damage_2015')
model_df = df.dropna(subset=key_cols).copy()

# Outcome: log1p deaths to reduce skew
model_df['log_deaths'] = np.log1p(model_df['deaths'])

# Controls: choose one damage measure to avoid collinearity; use 2013-adjusted if available
controls = ['category', 'min_pressure', 'max_wind']
if 'damage_2013' in model_df.columns:
    controls.append('damage_2013')
elif 'damage_2015' in model_df.columns:
    controls.append('damage_2015')

# Build design matrix
X = model_df[['masfem'] + controls]
X = sm.add_constant(X)

y = model_df['log_deaths']

# OLS regression with robust SEs (HC3)
ols_model = sm.OLS(y, X).fit(cov_type='HC3')

# Also check using binary female indicator as predictor
if 'female_binary' in model_df.columns:
    X2 = model_df[['female_binary'] + controls]
    X2 = sm.add_constant(X2)
    ols_model_bin = sm.OLS(y, X2).fit(cov_type='HC3')
else:
    ols_model_bin = None

# Simple correlation (Spearman) between masfem and deaths for robustness
spearman_corr = model_df[['masfem', 'deaths']].corr(method='spearman').iloc[0,1]

# Collect results
results = {
    'n': int(model_df.shape[0]),
    'ols_coef_masfem': float(ols_model.params['masfem']),
    'ols_p_masfem': float(ols_model.pvalues['masfem']),
    'ols_ci_masfem': [float(ols_model.conf_int().loc['masfem'][0]), float(ols_model.conf_int().loc['masfem'][1])],
    'ols_r2': float(ols_model.rsquared),
    'spearman_corr_masfem_deaths': float(spearman_corr)
}

if ols_model_bin is not None:
    results.update({
        'ols_coef_female_binary': float(ols_model_bin.params['female_binary']),
        'ols_p_female_binary': float(ols_model_bin.pvalues['female_binary']),
        'ols_ci_female_binary': [float(ols_model_bin.conf_int().loc['female_binary'][0]), float(ols_model_bin.conf_int().loc['female_binary'][1])],
        'ols_r2_female_binary': float(ols_model_bin.rsquared)
    })

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
