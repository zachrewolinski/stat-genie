import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Ensure expected columns exist
cols = df.columns.tolist()

# Basic info
n_rows = len(df)

# Variables
beauty = 'feature6'
rating = 'feature7'

# Drop rows with missing values in key vars
key_vars = [beauty, rating]
base_df = df.dropna(subset=key_vars).copy()

# Pearson correlation
corr, corr_p = stats.pearsonr(base_df[beauty], base_df[rating])

# Simple OLS
model_simple = smf.ols(f"{rating} ~ {beauty}", data=base_df).fit(cov_type='HC3')

# Multiple OLS with controls
# Categorical controls
controls = [
    'feature3',
    'C(feature4)',
    'C(feature2)',
    'C(feature5)',
    'C(feature8)',
    'C(feature9)',
    'C(feature10)',
    'feature11',
    'feature12'
]

formula = f"{rating} ~ {beauty} + " + " + ".join(controls)
model_controls = smf.ols(formula, data=df).fit(cov_type='HC3')

# Extract results
coef_simple = model_simple.params[beauty]
se_simple = model_simple.bse[beauty]
p_simple = model_simple.pvalues[beauty]

coef_controls = model_controls.params[beauty]
se_controls = model_controls.bse[beauty]
p_controls = model_controls.pvalues[beauty]

r2_simple = model_simple.rsquared
r2_controls = model_controls.rsquared

# Print summary for downstream reading
print({
    "n_rows": n_rows,
    "corr": corr,
    "corr_p": corr_p,
    "simple_coef": coef_simple,
    "simple_se": se_simple,
    "simple_p": p_simple,
    "simple_r2": r2_simple,
    "controls_coef": coef_controls,
    "controls_se": se_controls,
    "controls_p": p_controls,
    "controls_r2": r2_controls,
    "controls_n": int(model_controls.nobs),
})
