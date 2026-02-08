import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Basic cleaning
# Ensure children is categorical yes/no

df['children'] = df['children'].astype(str).str.lower()

# Create binary indicators

df['children_yes'] = (df['children'] == 'yes').astype(int)

df['any_affair'] = (df['affairs'] > 0).astype(int)

# Descriptive stats

group_stats = df.groupby('children')['affairs'].agg(['count', 'mean', 'median'])

# Proportion any affair

prop_any = df.groupby('children')['any_affair'].mean()

# OLS on affairs (counts)

ols_formula = 'affairs ~ children_yes + gender + age + yearsmarried + religiousness + education + occupation + rating'
ols_model = smf.ols(ols_formula, data=df).fit(cov_type='HC3')

# OLS on log(affairs+1)

log_df = df.copy()
log_df['log_affairs'] = np.log1p(log_df['affairs'])
log_model = smf.ols('log_affairs ~ children_yes + gender + age + yearsmarried + religiousness + education + occupation + rating', data=log_df).fit(cov_type='HC3')

# Logistic regression on any affair

logit_model = smf.logit('any_affair ~ children_yes + gender + age + yearsmarried + religiousness + education + occupation + rating', data=df).fit(disp=0)

# Extract key coefficients

ols_coef = ols_model.params['children_yes']
ols_p = ols_model.pvalues['children_yes']

log_coef = log_model.params['children_yes']
log_p = log_model.pvalues['children_yes']

logit_coef = logit_model.params['children_yes']
logit_p = logit_model.pvalues['children_yes']

# Build a simple scalar decision
# Negative coefficient implies children associated with fewer affairs -> supports "Yes" (decrease)
# Positive coefficient implies more affairs -> supports "No"
# If near zero/non-significant -> near neutral

# Use a score based on direction and significance of multiple models

score = 0.0

# OLS
if ols_p < 0.05:
    score += -40 if ols_coef < 0 else 40
else:
    score += -10 if ols_coef < 0 else 10

# Log OLS
if log_p < 0.05:
    score += -30 if log_coef < 0 else 30
else:
    score += -10 if log_coef < 0 else 10

# Logit
if logit_p < 0.05:
    score += -30 if logit_coef < 0 else 30
else:
    score += -10 if logit_coef < 0 else 10

# Clamp to [-100, 100]
score = max(-100, min(100, score))

# Print summary for review
print('Group stats (affairs mean):')
print(group_stats)
print('\nProportion any affair:')
print(prop_any)
print('\nOLS children_yes coef:', ols_coef, 'p=', ols_p)
print('Log OLS children_yes coef:', log_coef, 'p=', log_p)
print('Logit children_yes coef:', logit_coef, 'p=', logit_p)
print('\nScore:', score)
