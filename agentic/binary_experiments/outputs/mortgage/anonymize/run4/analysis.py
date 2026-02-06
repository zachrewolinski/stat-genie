import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# Outcome: 1 if accepted
if 'feature14' not in _df.columns:
    raise ValueError('Expected feature14 as acceptance indicator')

y = _df['feature14']

# Exclude the denial indicator to avoid perfect collinearity with acceptance
# Keep all other features as potential controls
X = _df.drop(columns=['feature14', 'feature11'], errors='ignore')

# Clean missing/inf values and add intercept
X = X.replace([np.inf, -np.inf], np.nan)
data = pd.concat([y, X], axis=1).dropna()
y = data['feature14']
X = data.drop(columns=['feature14'])
X = sm.add_constant(X, has_constant='add')

# Unadjusted acceptance rates by gender (feature2: 1 female, 0 male)
rate_by_gender = _df.groupby('feature2')['feature14'].mean().rename('accept_rate')

# Logistic regression controlling for other variables
# Use try/except to fall back to regularized fit if needed
try:
    model = sm.Logit(y, X).fit(disp=False)
    result = model
except Exception:
    model = sm.Logit(y, X)
    result = model.fit_regularized(disp=False)

# Extract coefficient and p-value for gender (feature2)
coef = result.params.get('feature2', np.nan)

# p-values are not available for regularized fit; handle gracefully
pval = None
if hasattr(result, 'pvalues'):
    pval = result.pvalues.get('feature2', np.nan)

print('Unadjusted acceptance rates by gender (feature2: 1 female, 0 male):')
print(rate_by_gender)
print('\nLogit coefficient for gender (feature2):', coef)
if pval is not None:
    print('P-value for gender (feature2):', pval)
else:
    print('P-value for gender (feature2): unavailable (regularized fit).')

# Save key outputs for downstream use
summary = {
    'accept_rate_male': rate_by_gender.get(0.0, np.nan),
    'accept_rate_female': rate_by_gender.get(1.0, np.nan),
    'gender_coef': coef,
    'gender_pvalue': pval,
    'n': len(_df),
}

pd.Series(summary).to_csv('analysis_summary.csv')
