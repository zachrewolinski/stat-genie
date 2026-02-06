import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('mortgage.csv')

n_rows = len(df)

# Identify binary columns
binary_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]

# Heuristic mapping based on metadata and distributions:
# - Column 'deny' appears to be acceptance indicator (mean ~0.88)
# - Column 'self_employed' is the inverse of 'deny' (mean ~0.12), likely denial indicator
# - Column 'denied_PMI' has mean ~0.21 and matches plausible female share
# - Column 'female' has mean ~0.02, plausible for PMI denial

gender_col = 'denied_PMI'
outcome_col = 'deny'  # acceptance indicator

# Basic sanity checks
if gender_col not in df.columns or outcome_col not in df.columns:
    raise ValueError('Expected columns not found in dataset.')

# Unadjusted acceptance rates by gender
ct = pd.crosstab(df[gender_col], df[outcome_col], normalize='index')
accept_rate_female = ct.loc[1, 1] if (1 in ct.index and 1 in ct.columns) else np.nan
accept_rate_male = ct.loc[0, 1] if (0 in ct.index and 1 in ct.columns) else np.nan
rate_diff = accept_rate_female - accept_rate_male

print('Unadjusted acceptance rates:')
print('female=1:', accept_rate_female)
print('female=0:', accept_rate_male)
print('difference:', rate_diff)

# Build regression controls: all numeric columns except outcome and obvious ID
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Drop outcome, denial indicator (perfect inverse), and high-uniqueness ID-like column
predictors = [c for c in numeric_cols if c != outcome_col]

# Drop the inverse of outcome if present
if 'self_employed' in predictors and (df['self_employed'] == (1 - df[outcome_col])).all():
    predictors.remove('self_employed')

# Drop obvious ID-like column (unique == n_rows)
for c in list(predictors):
    if df[c].nunique() == n_rows:
        predictors.remove(c)

# Drop any column with near-perfect collinearity (handled by statsmodels if needed)

X = df[predictors]
X = sm.add_constant(X, has_constant='add')

y = df[outcome_col]

model = sm.Logit(y, X, missing='drop')
result = model.fit(disp=False)

female_coef = result.params.get(gender_col, np.nan)
female_p = result.pvalues.get(gender_col, np.nan)

print('Logit coefficient for female (gender):', female_coef)
print('p-value:', female_p)

# Save key results to a simple dict for downstream use
results = {
    'accept_rate_female': float(accept_rate_female),
    'accept_rate_male': float(accept_rate_male),
    'rate_diff': float(rate_diff),
    'female_coef': float(female_coef),
    'female_p': float(female_p),
    'predictors_used': predictors,
}

pd.Series(results).to_csv('analysis_results.csv')
