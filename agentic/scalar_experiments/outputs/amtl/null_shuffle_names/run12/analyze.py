import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')
print('columns', _df.columns.tolist())
print(_df.head())
print('dtypes')
print(_df.dtypes)

# Basic summaries
for col in _df.columns:
    if _df[col].dtype == 'object':
        print(col, _df[col].nunique(), _df[col].unique()[:10])
    else:
        print(col, 'min', _df[col].min(), 'max', _df[col].max(), 'mean', _df[col].mean())

# Identify integer-like columns
for col in _df.columns:
    if _df[col].dtype != 'object':
        frac = (_df[col] - _df[col].round()).abs().mean()
        print('integer_like', col, frac)

# Candidate mapping
# Assume:
#   missing = genus
#   sockets = age
#   age_est = pop
#   sex_prob = stdev_age
#   tooth_class = sockets
#   genus_cat = tooth_class
#   population = specimen

missing = _df['genus']
possible_sockets = _df['age']
print('missing <= sockets proportion', (missing <= possible_sockets).mean())
print('missing==0 proportion', (missing==0).mean())
print('sockets values', possible_sockets.unique()[:20])

# Prepare model
# binomial GLM with response as successes/failures
# Add small check for bounds
valid = (missing >= 0) & (possible_sockets >= missing) & (possible_sockets > 0)
print('valid rows', valid.mean(), valid.sum())

_df2 = _df.loc[valid].copy()
_df2['missing'] = _df2['genus']
_df2['sockets_n'] = _df2['age']
_df2['age_est'] = _df2['pop']
_df2['sex_prob'] = _df2['stdev_age']
_df2['tooth_class_cat'] = _df2['sockets']
_df2['genus_cat'] = _df2['tooth_class']

# Encode categorical
# Use Homo sapiens as reference if present
print('genus categories', _df2['genus_cat'].unique())
print('tooth class categories', _df2['tooth_class_cat'].unique())

# Build design matrix
# Use patsy for binomial with endog as two-column array
import patsy

# Add predictors with Homo sapiens as reference for genus
formula = '1 + C(genus_cat, Treatment(reference=\"Homo sapiens\")) + age_est + sex_prob + C(tooth_class_cat)'

# Use success/failure
endog = np.column_stack([_df2['missing'], _df2['sockets_n'] - _df2['missing']])

X = patsy.dmatrix(formula, _df2, return_type='dataframe')
design_info = X.design_info

model = sm.GLM(endog, X, family=sm.families.Binomial())
res = model.fit()
print(res.summary())

print('params')
print(res.params)

# Marginal standardized predictions by genus
def predict_for_genus(genus_value: str) -> float:
    df_tmp = _df2.copy()
    df_tmp['genus_cat'] = genus_value
    X_tmp = patsy.build_design_matrices([design_info], df_tmp, return_type='dataframe')[0]
    preds = res.predict(X_tmp)
    return float(preds.mean())

genus_levels = sorted(_df2['genus_cat'].unique())
preds = {g: predict_for_genus(g) for g in genus_levels}
print('marginal_predictions', preds)

# Differences vs Homo sapiens
hs = preds['Homo sapiens']
diffs = {g: preds[g] - hs for g in genus_levels if g != 'Homo sapiens'}
print('diffs_vs_hs', diffs)

# Average observed rate by genus for context
_df2['rate'] = _df2['missing'] / _df2['sockets_n']
print('observed_rate_by_genus', _df2.groupby('genus_cat')['rate'].mean().to_dict())
