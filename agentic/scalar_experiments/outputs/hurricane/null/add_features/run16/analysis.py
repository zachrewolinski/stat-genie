import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Keep relevant columns and coerce numeric where needed
cols = [
    'masfem', 'gender_mf', 'alldeaths', 'ndam', 'ndam15',
    'wind', 'min', 'category', 'year', 'elapsedyrs'
]
missing_cols = [c for c in cols if c not in _df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

df = _df[cols].copy()

for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows with missing in key vars
base = df.dropna(subset=['masfem', 'alldeaths', 'wind', 'min', 'category']).copy()

# Transform outcomes
base['log1p_deaths'] = np.log1p(base['alldeaths'])
base['log1p_ndam15'] = np.log1p(base['ndam15'])

# Define models
models = {}
models['deaths_masfem_simple'] = smf.ols('log1p_deaths ~ masfem', data=base).fit(cov_type='HC3')
models['deaths_masfem_controls'] = smf.ols(
    'log1p_deaths ~ masfem + category + wind + min',
    data=base
).fit(cov_type='HC3')
models['deaths_gender_controls'] = smf.ols(
    'log1p_deaths ~ gender_mf + category + wind + min',
    data=base
).fit(cov_type='HC3')

# Add year control as robustness
models['deaths_masfem_controls_year'] = smf.ols(
    'log1p_deaths ~ masfem + category + wind + min + year',
    data=base
).fit(cov_type='HC3')

# Damage outcome as proxy for exposure/impact
models['damage_masfem_controls'] = smf.ols(
    'log1p_ndam15 ~ masfem + category + wind + min',
    data=base
).fit(cov_type='HC3')

# Correlations
corr_spearman = base[['masfem', 'alldeaths', 'ndam15', 'wind', 'min', 'category']].corr(method='spearman')

# Collect summary stats
summary = {
    'n_rows': int(base.shape[0]),
    'masfem_mean': float(base['masfem'].mean()),
    'alldeaths_mean': float(base['alldeaths'].mean()),
    'alldeaths_median': float(base['alldeaths'].median()),
}

# Extract key coefficients
results = {}
for name, m in models.items():
    params = m.params.to_dict()
    pvals = m.pvalues.to_dict()
    bse = m.bse.to_dict()
    results[name] = {
        'params': params,
        'pvals': pvals,
        'bse': bse,
        'r2': float(m.rsquared),
        'nobs': int(m.nobs),
    }

output = {
    'summary': summary,
    'corr_spearman': corr_spearman.round(4).to_dict(),
    'models': results,
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Wrote analysis_results.json')
