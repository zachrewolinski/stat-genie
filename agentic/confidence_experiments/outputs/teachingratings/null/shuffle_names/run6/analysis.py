import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

print('Columns:', list(_df.columns))
print(_df.head())
print('\nDtypes:')
print(_df.dtypes)

# Basic checks
print('\nMissing values per column:')
print(_df.isna().sum())

# Identify categorical columns (object or category)
cat_cols = [c for c in _df.columns if _df[c].dtype == 'object' or str(_df[c].dtype).startswith('category')]
print('\nCategorical columns:', cat_cols)
for c in cat_cols:
    print('\n', c, 'unique:', _df[c].unique()[:10], '... count', _df[c].nunique())

# Define outcome and predictor
outcome = 'allstudents'
predictor = 'beauty'

# Simple correlation
corr = _df[[outcome, predictor]].corr().iloc[0,1]
print('\nPearson correlation beauty vs allstudents:', corr)

# Simple OLS
model_simple = smf.ols(f"{outcome} ~ {predictor}", data=_df).fit()
print('\nSimple OLS summary:')
print(model_simple.summary())

# Build multivariate formula
# Use C() for categorical columns
# Exclude outcome and predictor from controls
controls = [c for c in _df.columns if c not in [outcome, predictor]]

# Build formula terms
terms = []
for c in controls:
    if c in cat_cols:
        terms.append(f"C({c})")
    else:
        terms.append(c)

formula = f"{outcome} ~ {predictor} + " + " + ".join(terms)
print('\nMultivariate formula:')
print(formula)

model_full = smf.ols(formula, data=_df).fit()
print('\nFull OLS summary (beauty coefficient):')
print(model_full.summary().tables[1])

# Extract key metrics
coef = model_full.params[predictor]
pval = model_full.pvalues[predictor]
conf_int = model_full.conf_int().loc[predictor].tolist()
print('\nFull model beauty coef:', coef)
print('p-value:', pval)
print('95% CI:', conf_int)

# Also compute standardized effect (beta)
# standardize outcome and predictor
z_df = _df.copy()
for col in [outcome, predictor]:
    z_df[col] = (z_df[col] - z_df[col].mean()) / z_df[col].std()
model_std = smf.ols(f"{outcome} ~ {predictor}", data=z_df).fit()
print('\nStandardized simple OLS coef (beta):', model_std.params[predictor])
print('p-value:', model_std.pvalues[predictor])
