import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure categorical columns are treated as categories
cat_cols = [
    'minority', 'gender', 'credits', 'division', 'native', 'tenure'
]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

# Drop rows with missing key variables
key_vars = ['eval', 'beauty']
for c in cat_cols:
    key_vars.append(c)
key_vars += ['age', 'students', 'allstudents']
key_vars = [c for c in key_vars if c in df.columns]

df_model = df.dropna(subset=key_vars).copy()

# Simple correlation
corr = df_model[['eval', 'beauty']].corr().iloc[0,1]

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=df_model).fit(cov_type='HC1')

# Multiple regression with controls
# Use C() for categoricals
formula_parts = ['beauty']
if 'age' in df_model.columns:
    formula_parts.append('age')
if 'students' in df_model.columns:
    formula_parts.append('students')
if 'allstudents' in df_model.columns:
    formula_parts.append('allstudents')
for c in cat_cols:
    if c in df_model.columns:
        formula_parts.append(f'C({c})')
formula = 'eval ~ ' + ' + '.join(formula_parts)
model_full = smf.ols(formula, data=df_model).fit(cov_type='HC1')

# Standardized effect for beauty in full model
# standardize eval and beauty (and continuous covariates) for comparability
cont_cols = ['eval', 'beauty']
for c in ['age', 'students', 'allstudents']:
    if c in df_model.columns:
        cont_cols.append(c)

df_std = df_model.copy()
for c in cont_cols:
    df_std[c] = (df_std[c] - df_std[c].mean()) / df_std[c].std(ddof=0)

formula_std = 'eval ~ ' + ' + '.join(formula_parts)
model_full_std = smf.ols(formula_std, data=df_std).fit(cov_type='HC1')

# Gather results
results = {
    'n': int(df_model.shape[0]),
    'corr_eval_beauty': float(corr),
    'simple_coef': float(model_simple.params.get('beauty', np.nan)),
    'simple_pvalue': float(model_simple.pvalues.get('beauty', np.nan)),
    'simple_ci': [float(x) for x in model_simple.conf_int().loc['beauty']]
        if 'beauty' in model_simple.params else [np.nan, np.nan],
    'full_coef': float(model_full.params.get('beauty', np.nan)),
    'full_pvalue': float(model_full.pvalues.get('beauty', np.nan)),
    'full_ci': [float(x) for x in model_full.conf_int().loc['beauty']]
        if 'beauty' in model_full.params else [np.nan, np.nan],
    'full_std_coef': float(model_full_std.params.get('beauty', np.nan)),
    'full_r2': float(model_full.rsquared),
    'simple_r2': float(model_simple.rsquared),
}

print(json.dumps(results, indent=2))
