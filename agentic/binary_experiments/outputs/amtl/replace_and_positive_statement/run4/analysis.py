import math
import pandas as pd
import statsmodels.api as sm
import patsy

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing critical fields
_df = _df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus'])

# Create binary indicator for Homo sapiens vs non-human primates
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Successes and failures for binomial GLM
_df['failures'] = _df['sockets'] - _df['num_amtl']

# Build design matrix
X = patsy.dmatrix('is_human + age + prob_male + C(tooth_class)', data=_df, return_type='dataframe')
endog = _df[['num_amtl', 'failures']]

# Fit binomial GLM with logit link
model = sm.GLM(endog, X, family=sm.families.Binomial())
result = model.fit()

# Extract coefficient and p-value for is_human
coef = result.params['is_human']
pval = result.pvalues['is_human']

# Odds ratio
odds_ratio = math.exp(coef)

print(result.summary())
print('\nKey result:')
print(f'is_human coef (log-odds): {coef:.4f}')
print(f'is_human odds ratio: {odds_ratio:.4f}')
print(f'is_human p-value: {pval:.6g}')
