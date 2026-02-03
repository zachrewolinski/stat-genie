import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Basic derived variables
# Binary: any affair
_df = df.copy()
_df['affair_any'] = (_df['affairs'] > 0).astype(int)
_df['children_yes'] = (_df['children'].str.lower() == 'yes').astype(int)

# Descriptive comparisons
mean_by_children = _df.groupby('children')['affairs'].mean()
prop_any_by_children = _df.groupby('children')['affair_any'].mean()

print('Mean affairs by children status:')
print(mean_by_children)
print('\nProportion with any affairs by children status:')
print(prop_any_by_children)

# Difference in means (children yes - no)
mean_diff = mean_by_children.get('yes', np.nan) - mean_by_children.get('no', np.nan)
prop_diff = prop_any_by_children.get('yes', np.nan) - prop_any_by_children.get('no', np.nan)
print('\nDifference (yes - no):')
print({'mean_affairs_diff': mean_diff, 'prop_any_diff': prop_diff})

# OLS with controls
ols_formula = (
    'affairs ~ children_yes + C(gender) + age + yearsmarried + '
    'religiousness + education + occupation + rating'
)
ols_model = smf.ols(ols_formula, data=_df).fit()
print('\nOLS results (affairs count):')
print(ols_model.summary().tables[1])

# Logit for any affair
logit_formula = (
    'affair_any ~ children_yes + C(gender) + age + yearsmarried + '
    'religiousness + education + occupation + rating'
)
logit_model = smf.logit(logit_formula, data=_df).fit(disp=False)
print('\nLogit results (any affair):')
print(logit_model.summary().tables[1])

# Extract key coefficients
print('\nKey coefficients:')
print('OLS children_yes coef:', ols_model.params['children_yes'])
print('OLS children_yes p-value:', ols_model.pvalues['children_yes'])
print('Logit children_yes coef:', logit_model.params['children_yes'])
print('Logit children_yes p-value:', logit_model.pvalues['children_yes'])
