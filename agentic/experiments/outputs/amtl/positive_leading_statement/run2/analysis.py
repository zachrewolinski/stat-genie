import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing required fields
required_cols = [
    'num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus'
]
_df = _df.dropna(subset=required_cols).copy()

# Binary indicator: modern human vs non-human primates
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Ensure valid counts
_df = _df[(_df['sockets'] > 0) & (_df['num_amtl'] >= 0) & (_df['num_amtl'] <= _df['sockets'])]

_df['num_nonamtl'] = _df['sockets'] - _df['num_amtl']

# Design matrices
formula = 'is_human + age + prob_male + C(tooth_class)'
X = patsy.dmatrix(formula, data=_df, return_type='dataframe')
y = np.column_stack((_df['num_amtl'].to_numpy(), _df['num_nonamtl'].to_numpy()))

model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

print(model.summary())

# Extract effect for is_human
coef = model.params.get('is_human', float('nan'))
pval = model.pvalues.get('is_human', float('nan'))

# Convert to odds ratio for interpretability
odds_ratio = float('nan')
if pd.notna(coef):
    odds_ratio = float(np.exp(coef))

print(f"is_human coef: {coef:.6f}")
print(f"is_human p-value: {pval:.6g}")
print(f"is_human odds ratio: {odds_ratio:.6f}")
