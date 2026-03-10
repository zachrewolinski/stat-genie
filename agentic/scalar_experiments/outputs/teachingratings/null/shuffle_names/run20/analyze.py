import json
import pandas as pd
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Identify candidate columns
print('Columns:', df.columns.tolist())
print(df.head())

# Basic stats
print('\nDtypes:')
print(df.dtypes)

# Assume beauty and allstudents columns for analysis
if 'beauty' not in df.columns or 'allstudents' not in df.columns:
    raise SystemExit('Expected beauty and allstudents columns not found')

# Remove rows with missing values in key columns
key_df = df[['beauty','allstudents']].dropna()
print('\nN for beauty/allstudents:', len(key_df))

# Correlation
corr = key_df['beauty'].corr(key_df['allstudents'])
print('Correlation beauty vs allstudents:', corr)

# Simple linear regression
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()
print('\nSimple regression summary:')
print(model_simple.summary())

# Build multivariate model with other columns as controls
# Select all columns except outcome
control_cols = [c for c in df.columns if c not in ['allstudents']]

# Build formula: allstudents ~ beauty + C(cat) ...
terms = []
for col in control_cols:
    if col == 'allstudents':
        continue
    if df[col].dtype == 'object':
        terms.append(f'C({col})')
    else:
        terms.append(col)

formula = 'allstudents ~ ' + ' + '.join(terms)
print('\nFormula:', formula)

model_full = smf.ols(formula, data=df).fit()
print('\nFull regression summary:')
print(model_full.summary())

# Save key stats for later use
results = {
    'corr': corr,
    'simple_coef': model_simple.params.get('beauty'),
    'simple_p': model_simple.pvalues.get('beauty'),
    'simple_r2': model_simple.rsquared,
    'full_coef': model_full.params.get('beauty'),
    'full_p': model_full.pvalues.get('beauty'),
    'full_r2': model_full.rsquared,
    'n': int(model_simple.nobs)
}

with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print('\nSaved analysis_results.json')
