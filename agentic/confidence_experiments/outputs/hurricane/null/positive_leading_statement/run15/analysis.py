import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')

# Basic transforms
# Add small constant for log transform
# Deaths are counts; use log1p to reduce skew

df['log_deaths'] = np.log1p(df['alldeaths'])

# Standardize some severity measures for interpretability in interaction
for col in ['masfem', 'wind', 'min', 'category', 'ndam15']:
    if col in df.columns:
        df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

models = {}

# Simple bivariate
models['m1'] = smf.ols('log_deaths ~ z_masfem', data=df).fit(cov_type='HC3')

# Controls: wind, pressure, category
models['m2'] = smf.ols('log_deaths ~ z_masfem + z_wind + z_min + z_category', data=df).fit(cov_type='HC3')

# Controls alternative: wind + category (drop min to reduce collinearity)
models['m3'] = smf.ols('log_deaths ~ z_masfem + z_wind + z_category', data=df).fit(cov_type='HC3')

# Interaction with storm intensity (wind)
models['m4'] = smf.ols('log_deaths ~ z_masfem * z_wind + z_category', data=df).fit(cov_type='HC3')

# Interaction with pressure
models['m5'] = smf.ols('log_deaths ~ z_masfem * z_min + z_category', data=df).fit(cov_type='HC3')

# Optional: using MTurk femininity rating
if 'masfem_mturk' in df.columns:
    df['z_masfem_mturk'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / df['masfem_mturk'].std(ddof=0)
    models['m6'] = smf.ols('log_deaths ~ z_masfem_mturk + z_wind + z_category', data=df).fit(cov_type='HC3')

# Collect key stats
summary_rows = []
for name, model in models.items():
    for term in model.params.index:
        if term.startswith('z_masfem'):
            summary_rows.append({
                'model': name,
                'term': term,
                'coef': model.params[term],
                'pvalue': model.pvalues[term],
                'ci_low': model.conf_int().loc[term, 0],
                'ci_high': model.conf_int().loc[term, 1],
                'n': int(model.nobs),
                'r2': model.rsquared
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('analysis_summary.csv', index=False)

# Also save model summaries for inspection
with open('model_summaries.txt', 'w') as f:
    for name, model in models.items():
        f.write(f'=== {name} ===\n')
        f.write(model.summary().as_text())
        f.write('\n\n')

print(summary_df)
