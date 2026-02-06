import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv('affairs.csv')

# Map columns for readability
DF = DF.rename(columns={
    'feature2': 'affairs_freq',
    'feature3': 'gender',
    'feature4': 'age',
    'feature5': 'years_married',
    'feature6': 'children',
    'feature7': 'religiosity',
    'feature8': 'education',
    'feature9': 'occupation',
    'feature10': 'marriage_rating'
})

# Binary indicator for any affair
DF['any_affair'] = (DF['affairs_freq'] > 0).astype(int)

# Descriptive stats by children
summary = DF.groupby('children').agg(
    n=('affairs_freq', 'size'),
    mean_affairs=('affairs_freq', 'mean'),
    median_affairs=('affairs_freq', 'median'),
    prop_any=('any_affair', 'mean')
).reset_index()

print('Summary by children')
print(summary.to_string(index=False))

# Logistic regression for any affair with controls
# Use categorical encoding for gender and children
logit_formula = (
    'any_affair ~ C(children) + C(gender) + age + years_married + '
    'religiosity + education + occupation + marriage_rating'
)
logit_model = smf.logit(logit_formula, data=DF).fit(disp=False)
print('\nLogit coefficients')
print(logit_model.summary2().tables[1])

# OLS on log(1 + affairs frequency) with same controls
DF['log_affairs'] = np.log1p(DF['affairs_freq'])
ols_formula = (
    'log_affairs ~ C(children) + C(gender) + age + years_married + '
    'religiosity + education + occupation + marriage_rating'
)
ols_model = smf.ols(ols_formula, data=DF).fit()
print('\nOLS coefficients')
print(ols_model.summary2().tables[1])

# Save key results for later reference
summary.to_csv('analysis_summary.csv', index=False)
logit_model.summary2().tables[1].to_csv('analysis_logit_coefs.csv')
ols_model.summary2().tables[1].to_csv('analysis_ols_coefs.csv')
