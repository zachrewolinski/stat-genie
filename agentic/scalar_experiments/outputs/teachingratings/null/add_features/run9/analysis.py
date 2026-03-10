import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'

df = pd.read_csv(csv_path)

# Ensure columns
# Standardize column names maybe repeated? There is gender duplicate? let's inspect.

print('columns', df.columns.tolist())
print('shape', df.shape)

# Basic summary

# We'll focus on beauty and eval
# Drop rows with missing values in these
sub = df[['beauty','eval']].copy()
sub = sub.dropna()
print('non-missing', sub.shape)

# Correlation
corr = sub['beauty'].corr(sub['eval'])
print('corr', corr)

# Simple linear regression
model_simple = smf.ols('eval ~ beauty', data=df).fit()
print(model_simple.summary())

# Multivariate regression with typical controls
# Identify available variables from dataset (may have extra features from add_features). We'll use those that are plausible controls
# Use categorical variables with C()

# candidate controls in dataset
controls = []
for col in ['age','gender','minority','native','tenure','division','credits','students','allstudents']:
    if col in df.columns:
        controls.append(col)

# Build formula
formula = 'eval ~ beauty'
for col in controls:
    if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
        formula += f' + C({col})'
    else:
        formula += f' + {col}'

model_controls = smf.ols(formula, data=df).fit()
print('formula', formula)
print(model_controls.summary())

# Additional: standardized effect size (beta) for beauty in simple and controls

def standardized_beta(model, df, predictor, response='eval'):
    # standardize columns and refit OLS
    cols = [response] + [predictor]
    # if model has more predictors, we can compute by standardizing all numeric and using same formula
    # But we will compute for simple model
    d = df[[response, predictor]].dropna()
    d_std = (d - d.mean()) / d.std(ddof=0)
    m = smf.ols(f'{response} ~ {predictor}', data=d_std).fit()
    return m.params[predictor]

beta_simple = standardized_beta(model_simple, df, 'beauty', 'eval')
print('beta_simple', beta_simple)

# For controlled model, compute standardized beta by standardizing numeric predictors.

# Build standardized dataframe for controlled model
ctrl_cols = []
for col in controls:
    ctrl_cols.append(col)

# separate numeric vs categorical
num_cols = [c for c in ctrl_cols if (pd.api.types.is_numeric_dtype(df[c]))]
cat_cols = [c for c in ctrl_cols if c not in num_cols]

# Use df for model
model_df = df[['eval','beauty'] + num_cols + cat_cols].dropna()

# Standardize numeric columns
for col in ['eval','beauty'] + num_cols:
    model_df[col] = (model_df[col] - model_df[col].mean()) / model_df[col].std(ddof=0)

# Build formula with categories
formula_std = 'eval ~ beauty'
for col in num_cols:
    formula_std += f' + {col}'
for col in cat_cols:
    formula_std += f' + C({col})'

model_std = smf.ols(formula_std, data=model_df).fit()
print('formula_std', formula_std)
print(model_std.params.get('beauty'))

# Save key results to json
results = {
    'n': int(len(sub)),
    'corr': float(corr),
    'simple_coef': float(model_simple.params['beauty']),
    'simple_p': float(model_simple.pvalues['beauty']),
    'simple_r2': float(model_simple.rsquared),
    'controls_formula': formula,
    'controls_coef': float(model_controls.params['beauty']),
    'controls_p': float(model_controls.pvalues['beauty']),
    'controls_r2': float(model_controls.rsquared),
    'std_beta_simple': float(beta_simple),
    'std_beta_controls': float(model_std.params.get('beauty', np.nan)),
    'n_controls': int(model_controls.nobs)
}

with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print('saved analysis_results.json')
