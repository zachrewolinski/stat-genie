import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
_df = pd.read_csv('mortgage.csv')

# Identify outcome column: use 'accept' as approval indicator
# The 'deny' column is not a clean complement of 'accept' in this dataset.
if 'accept' not in _df.columns:
    raise ValueError("Expected 'accept' column for approval outcome.")

# Compute raw approval rates by gender
if 'female' not in _df.columns:
    raise ValueError("Expected 'female' column for gender indicator.")

approval_by_gender = _df.groupby('female')['accept'].mean()
print('Approval rate by gender (female=1, male=0):')
print(approval_by_gender)

# Logistic regression controlling for other observed covariates
features = [c for c in _df.columns if c != 'accept']
X = _df[features].select_dtypes(include=[np.number]).copy()

# Remove columns with no variation
X = X.loc[:, X.nunique(dropna=True) > 1]

# Assemble modeling dataset
model_data = pd.concat([_df['accept'], X], axis=1).dropna()
y = model_data['accept']
X = model_data.drop(columns=['accept'])
X = sm.add_constant(X, has_constant='add')

model = sm.Logit(y, X).fit(disp=False)
print('\nLogit regression results (coefficients):')
print(model.summary().tables[1])

if 'female' in model.params.index:
    coef = model.params['female']
    pval = model.pvalues['female']
    print(f"\nFemale coefficient: {coef:.4f}, p-value: {pval:.4f}")
else:
    print("\nFemale coefficient not found in model.")
