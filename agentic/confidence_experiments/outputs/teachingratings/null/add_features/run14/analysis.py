import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"

df = pd.read_csv(path)

# Basic cleaning: ensure column names are unique (if duplicates) by de-duplicating
# Pandas will auto-mangle duplicates with .1, but enforce if necessary

# Identify target columns
# Expect columns: beauty, eval, plus controls

# Print columns for debugging (optional)
# print(df.columns)

# Ensure numeric

# We'll attempt analysis with available columns

# Drop rows with missing beauty or eval

if 'beauty' not in df.columns or 'eval' not in df.columns:
    raise ValueError("Expected columns 'beauty' and 'eval' not found")

# Keep only needed columns for regression

# Select control variables if present
controls = []
for col in ['age', 'gender', 'minority', 'native', 'tenure', 'division', 'credits', 'students', 'allstudents', 'female']:
    if col in df.columns:
        controls.append(col)

# Some columns are categorical; handle with formula and C()

analysis_df = df.copy()

# Prepare separate dataframes for simple and controlled analyses
simple_df = analysis_df[['beauty', 'eval']].dropna()

# Compute correlation
corr = simple_df['beauty'].corr(simple_df['eval'])

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=simple_df).fit()

# Multiple OLS with controls if any
if controls:
    use_cols = ['beauty', 'eval'] + controls
    ctrl_df = analysis_df[use_cols].dropna()
    # Build formula
    formula_parts = ['beauty']
    for col in controls:
        if analysis_df[col].dtype == 'O' or analysis_df[col].dtype.name == 'category':
            formula_parts.append(f'C({col})')
        else:
            formula_parts.append(col)
    formula = 'eval ~ ' + ' + '.join(formula_parts)
    model_controls = smf.ols(formula, data=ctrl_df).fit()
else:
    model_controls = None

# Extract results
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']

if model_controls is not None and 'beauty' in model_controls.params:
    ctrl_coef = model_controls.params['beauty']
    ctrl_p = model_controls.pvalues['beauty']
else:
    ctrl_coef = None
    ctrl_p = None

# Build narrative

results = {
    'n_simple': int(len(simple_df)),
    'n_controls': int(len(ctrl_df)) if controls else None,
    'corr': float(corr),
    'simple_coef': float(simple_coef),
    'simple_p': float(simple_p),
    'ctrl_coef': float(ctrl_coef) if ctrl_coef is not None else None,
    'ctrl_p': float(ctrl_p) if ctrl_p is not None else None,
    'controls': controls,
}

# Save intermediate results
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
