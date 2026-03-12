import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Rename for readability
cols = {
    'feature4': 'masfem',
    'feature6': 'female',
    'feature8': 'deaths',
    'feature5': 'min_pressure',
    'feature7': 'category',
    'feature13': 'max_wind',
    'feature2': 'year',
}
_df = _df.rename(columns=cols)

# Derived variables
_df['log_deaths'] = np.log1p(_df['deaths'])

# Standardize continuous predictors for comparability
for c in ['masfem', 'min_pressure', 'max_wind', 'category', 'year']:
    if c in _df:
        _df[c + '_z'] = (_df[c] - _df[c].mean()) / _df[c].std(ddof=0)

# Helper to fit OLS with HC3 SEs

def fit_ols(y, X, data):
    X = sm.add_constant(data[X])
    model = sm.OLS(data[y], X).fit(cov_type='HC3')
    return model

results = {}

# Model 1: log_deaths ~ masfem (continuous)
results['m1'] = fit_ols('log_deaths', ['masfem_z'], _df)

# Model 2: log_deaths ~ masfem + severity controls (pressure, wind, category)
results['m2'] = fit_ols('log_deaths', ['masfem_z', 'min_pressure_z', 'max_wind_z', 'category_z'], _df)

# Model 3: add year (proxy for preparedness improvements)
results['m3'] = fit_ols('log_deaths', ['masfem_z', 'min_pressure_z', 'max_wind_z', 'category_z', 'year_z'], _df)

# Model 4: binary female name instead of continuous masfem
results['m4'] = fit_ols('log_deaths', ['female', 'min_pressure_z', 'max_wind_z', 'category_z', 'year_z'], _df)

# Simple correlation
corr = _df[['masfem', 'deaths']].corr().iloc[0,1]

# Output key stats
print('N:', len(_df))
print('Correlation masfem vs deaths:', corr)

for k, m in results.items():
    coef = m.params.get('masfem_z', np.nan)
    pval = m.pvalues.get('masfem_z', np.nan)
    if k == 'm4':
        coef = m.params.get('female', np.nan)
        pval = m.pvalues.get('female', np.nan)
        print(f"{k}: coef(female)={coef:.4f}, p={pval:.4f}, R2={m.rsquared:.3f}")
    else:
        print(f"{k}: coef(masfem_z)={coef:.4f}, p={pval:.4f}, R2={m.rsquared:.3f}")

# Save summary for reference
summary = {
    'corr_masfem_deaths': float(corr),
    'models': {}
}
for k, m in results.items():
    if k == 'm4':
        coef = m.params.get('female', np.nan)
        pval = m.pvalues.get('female', np.nan)
    else:
        coef = m.params.get('masfem_z', np.nan)
        pval = m.pvalues.get('masfem_z', np.nan)
    summary['models'][k] = {
        'coef': float(coef),
        'pval': float(pval),
        'r2': float(m.rsquared),
        'n': int(m.nobs),
    }

pd.Series(summary).to_json('analysis_summary.json')
