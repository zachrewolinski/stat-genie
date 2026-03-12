import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic cleaning: ensure columns exist
print('Columns:', df.columns.tolist())

# Drop rows with missing beauty or eval
sub = df[['beauty', 'eval']].dropna()
print('N total:', len(df), 'N with beauty+eval:', len(sub))

# Correlation
corr = sub['beauty'].corr(sub['eval'])
print('Correlation beauty-eval:', corr)

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit()
print('\nSimple OLS summary:')
print(model_simple.summary())

# Add controls if present
# Potential control columns
control_cols = ['age', 'gender', 'division', 'native', 'tenure', 'students', 'credits', 'minority']
existing_controls = [c for c in control_cols if c in df.columns]
print('Existing controls:', existing_controls)

if existing_controls:
    # Build formula with categorical handling
    formula = 'eval ~ beauty'
    for c in existing_controls:
        if df[c].dtype == 'object':
            formula += f' + C({c})'
        else:
            formula += f' + {c}'
    model_controls = smf.ols(formula, data=df).fit()
    print('\nOLS with controls summary:')
    print(model_controls.summary())

# Robust SE (HC3) for the simple model
robust = model_simple.get_robustcov_results(cov_type='HC3')
print('\nSimple OLS with HC3 SE:')
print(robust.summary())
