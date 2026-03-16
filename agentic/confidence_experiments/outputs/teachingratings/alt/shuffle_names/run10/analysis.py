import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('teachingratings.csv')

# Basic info
print('rows', len(df))
print('columns', df.columns.tolist())

# Ensure numeric columns
# allstudents and beauty should be numeric
for col in ['beauty', 'allstudents']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing key variables
key_df = df[['beauty', 'allstudents']].dropna()
print('non-missing beauty/allstudents', len(key_df))

# Correlations
pearson_r, pearson_p = stats.pearsonr(key_df['beauty'], key_df['allstudents'])
spearman_r, spearman_p = stats.spearmanr(key_df['beauty'], key_df['allstudents'])
print('pearson_r', pearson_r, 'pearson_p', pearson_p)
print('spearman_r', spearman_r, 'spearman_p', spearman_p)

# Simple linear regression
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()
print('\nSimple OLS')
print(model_simple.summary())

# Build controls: treat categorical columns as categorical
# Identify potential categorical columns by dtype or unique count
categorical_cols = []
for col in df.columns:
    if col in ['beauty', 'allstudents']:
        continue
    if df[col].dtype == object:
        categorical_cols.append(col)

# For numeric columns, keep as-is

# Build formula with all other columns as controls
control_terms = []
for col in df.columns:
    if col in ['beauty', 'allstudents']:
        continue
    if col in categorical_cols:
        control_terms.append(f'C({col})')
    else:
        control_terms.append(col)

formula = 'allstudents ~ beauty'
if control_terms:
    formula += ' + ' + ' + '.join(control_terms)

model_full = smf.ols(formula, data=df).fit()
print('\nFull OLS with controls')
print('formula', formula)
print(model_full.summary())

# Extract coefficient and CI for beauty in full model
coef = model_full.params.get('beauty', np.nan)
se = model_full.bse.get('beauty', np.nan)
ci_low, ci_high = model_full.conf_int().loc['beauty']
print('\nbeauty_coef_full', coef)
print('beauty_se_full', se)
print('beauty_ci_full', (ci_low, ci_high))
print('beauty_p_full', model_full.pvalues.get('beauty', np.nan))
