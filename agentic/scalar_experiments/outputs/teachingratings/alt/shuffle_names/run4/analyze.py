import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic checks
print('Rows:', len(_df))
print('Columns:', list(_df.columns))

# Outcome and predictor
outcome = 'allstudents'
predictor = 'beauty'

# Descriptive stats
print('\nDescriptive stats:')
print(_df[[outcome, predictor]].describe())

# Pearson correlation
r, p = stats.pearsonr(_df[predictor], _df[outcome])
print(f"\nPearson r between {predictor} and {outcome}: {r:.4f}, p={p:.4g}")

# Simple OLS
model_simple = smf.ols(f"{outcome} ~ {predictor}", data=_df).fit(cov_type='HC3')
print('\nSimple OLS (HC3 robust):')
print(model_simple.summary().tables[1])

# Build multivariate model with controls
# Identify categorical columns (object dtype)
cat_cols = [c for c in _df.columns if _df[c].dtype == 'object']

# Exclude high-cardinality identifiers and non-informative ids
exclude_cols = {outcome, predictor, 'division', 'students'}

# Numeric controls
num_cols = [c for c in _df.columns if c not in exclude_cols and _df[c].dtype != 'object']

# Categorical controls
cat_controls = [c for c in cat_cols if c not in exclude_cols]

# Build formula
controls = num_cols + [f"C({c})" for c in cat_controls]
formula = f"{outcome} ~ {predictor}"
if controls:
    formula += " + " + " + ".join(controls)

print('\nMultivariate formula:')
print(formula)

model_full = smf.ols(formula, data=_df).fit(cov_type='HC3')
print('\nMultivariate OLS (HC3 robust):')
print(model_full.summary().tables[1])

# Extract key stats for beauty
coef_simple = model_simple.params[predictor]
se_simple = model_simple.bse[predictor]
p_simple = model_simple.pvalues[predictor]

coef_full = model_full.params[predictor]
se_full = model_full.bse[predictor]
p_full = model_full.pvalues[predictor]

# Effect size per 1 SD beauty
sd_beauty = _df[predictor].std()
impact_simple = coef_simple * sd_beauty
impact_full = coef_full * sd_beauty

print('\nKey stats:')
print({
    'coef_simple': coef_simple,
    'se_simple': se_simple,
    'p_simple': p_simple,
    'coef_full': coef_full,
    'se_full': se_full,
    'p_full': p_full,
    'sd_beauty': sd_beauty,
    'impact_simple_1sd': impact_simple,
    'impact_full_1sd': impact_full,
})
