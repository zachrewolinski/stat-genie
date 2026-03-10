import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic prep
# log deaths to reduce skew
for col in ['alldeaths','ndam','ndam15']:
    if col in df.columns:
        df[f'log_{col}'] = np.log1p(df[col])

# Quick summaries
summary = {
    'n': len(df),
    'female_count': df['gender_mf'].sum(),
    'male_count': (df['gender_mf'] == 0).sum(),
    'deaths_mean': df['alldeaths'].mean(),
    'deaths_median': df['alldeaths'].median(),
}

# Correlations
corrs = {
    'masfem_log_deaths_corr': df['masfem'].corr(df['log_alldeaths']),
    'masfem_deaths_corr': df['masfem'].corr(df['alldeaths']),
}

# Simple group comparison: female vs male
female = df[df['gender_mf'] == 1]['log_alldeaths']
male = df[df['gender_mf'] == 0]['log_alldeaths']
mean_diff = female.mean() - male.mean()

# Regression models
# Baseline controls: wind, min pressure, category, damage, year
# To avoid multicollinearity, we can try different specs.

models = {}

# 1) log deaths ~ masfem
models['m1'] = smf.ols('log_alldeaths ~ masfem', data=df).fit()

# 2) log deaths ~ masfem + category + wind + min
models['m2'] = smf.ols('log_alldeaths ~ masfem + category + wind + min', data=df).fit()

# 3) add damage and year
models['m3'] = smf.ols('log_alldeaths ~ masfem + category + wind + min + log_ndam15 + year', data=df).fit()

# 4) use gender_mf instead of masfem
models['m4'] = smf.ols('log_alldeaths ~ gender_mf + category + wind + min + log_ndam15 + year', data=df).fit()

# Output key results
results = {}
for k, m in models.items():
    coef = m.params.get('masfem', np.nan)
    pval = m.pvalues.get('masfem', np.nan)
    coef_g = m.params.get('gender_mf', np.nan)
    pval_g = m.pvalues.get('gender_mf', np.nan)
    results[k] = {
        'r2': m.rsquared,
        'masfem_coef': coef,
        'masfem_pval': pval,
        'gender_mf_coef': coef_g,
        'gender_mf_pval': pval_g,
        'n': int(m.nobs),
    }

# Compute effect sizes: percent change in deaths for 1-unit masfem (approx exp coef -1)
# For log1p, approximate by exp(coef)-1
for k, m in models.items():
    if 'masfem' in m.params:
        results[k]['masfem_pct'] = float(np.expm1(m.params['masfem']))
    if 'gender_mf' in m.params:
        results[k]['gender_mf_pct'] = float(np.expm1(m.params['gender_mf']))

print('SUMMARY', summary)
print('CORRS', corrs)
print('MEAN_DIFF_LOG_DEATHS_FEMALE_MINUS_MALE', mean_diff)
for k in sorted(results):
    print(k, results[k])

# Save a small table for inspection
out = pd.DataFrame({
    'name': df['name'],
    'year': df['year'],
    'masfem': df['masfem'],
    'gender_mf': df['gender_mf'],
    'alldeaths': df['alldeaths'],
    'log_alldeaths': df['log_alldeaths'],
    'category': df['category'],
    'wind': df['wind'],
    'min': df['min'],
    'log_ndam15': df['log_ndam15'],
})
print('HEAD')
print(out.head().to_string(index=False))

