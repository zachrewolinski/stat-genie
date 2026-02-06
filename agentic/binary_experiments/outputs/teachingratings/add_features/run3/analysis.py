import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('teachingratings.csv')

# Keep relevant columns for the research question
cols = [
    'eval', 'beauty', 'age', 'students', 'allstudents',
    'minority', 'gender', 'credits', 'division', 'native', 'tenure'
]

# Drop rows with missing data in relevant columns
_df = _df[cols].dropna().copy()

# One-hot encode categorical variables
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
X = pd.get_dummies(_df.drop(columns=['eval']), columns=cat_cols, drop_first=True)

# Add constant
X = sm.add_constant(X)

# Fit OLS regression with robust standard errors
model = sm.OLS(_df['eval'], X).fit(cov_type='HC3')

# Output key results
coef = model.params.get('beauty', float('nan'))
pval = model.pvalues.get('beauty', float('nan'))

print('N:', int(model.nobs))
print('Beauty coefficient:', coef)
print('Beauty p-value:', pval)
print(model.summary())
