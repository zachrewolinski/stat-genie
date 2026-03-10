import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('teachingratings.csv')

# Ensure numeric columns
for col in df.columns:
    if df[col].dtype == 'object':
        # keep as object for formula (categorical)
        continue
    # Attempt to coerce to numeric
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Basic stats
n = len(df)

# target variables
beauty = df['beauty']
ratings = df['allstudents']

# Drop missing
mask = beauty.notna() & ratings.notna()
beauty = beauty[mask]
ratings = ratings[mask]

# Pearson correlation
pearson_r, pearson_p = stats.pearsonr(beauty, ratings)

# Spearman correlation
spearman_r, spearman_p = stats.spearmanr(beauty, ratings)

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()

# Build control formula: exclude outcome and beauty; drop near-unique columns to avoid IDs
n_unique = df.nunique(dropna=True)

control_cols = []
for col in df.columns:
    if col in ('allstudents', 'beauty'):
        continue
    # Exclude columns that are almost unique (likely identifiers)
    if n_unique[col] >= 0.9 * n:
        continue
    control_cols.append(col)

# Build formula with categorical handling for object dtype columns
terms = []
for col in control_cols:
    if df[col].dtype == 'object':
        terms.append(f'C({col})')
    else:
        terms.append(col)

formula_controls = 'allstudents ~ beauty'
if terms:
    formula_controls += ' + ' + ' + '.join(terms)

model_controls = smf.ols(formula_controls, data=df).fit()

# Collect key metrics
results = {
    'n': int(n),
    'pearson_r': float(pearson_r),
    'pearson_p': float(pearson_p),
    'spearman_r': float(spearman_r),
    'spearman_p': float(spearman_p),
    'simple_slope': float(model_simple.params['beauty']),
    'simple_p': float(model_simple.pvalues['beauty']),
    'simple_ci_low': float(model_simple.conf_int().loc['beauty'][0]),
    'simple_ci_high': float(model_simple.conf_int().loc['beauty'][1]),
    'simple_r2': float(model_simple.rsquared),
    'controls_formula': formula_controls,
    'controls_slope': float(model_controls.params['beauty']),
    'controls_p': float(model_controls.pvalues['beauty']),
    'controls_ci_low': float(model_controls.conf_int().loc['beauty'][0]),
    'controls_ci_high': float(model_controls.conf_int().loc['beauty'][1]),
    'controls_r2': float(model_controls.rsquared),
    'control_cols': control_cols,
    'n_unique': {k: int(v) for k, v in n_unique.items()},
}

print(json.dumps(results, indent=2))
