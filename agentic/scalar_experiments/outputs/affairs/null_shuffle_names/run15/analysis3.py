import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns based on observed distributions
# affairs count (0,1,2,3,7,12) appears in column 'age'
_df['affairs_count'] = _df['age']
# children indicator appears in column 'religiousness' (yes/no)
_df['children_yes'] = (_df['religiousness'].str.lower() == 'yes').astype(int)

# Other covariates mapped by distributions
_df['years_married'] = _df['children']  # 0.125..15
_df['age_years'] = _df['occupation']    # 17.5..57
_df['education_years'] = _df['yearsmarried']  # 9..20
_df['occupation_code'] = _df['rownames']      # 1..7
_df['religiousness_scale'] = _df['rating']    # 1..5
_df['marriage_rating'] = _df['affairs']       # 1..5
_df['gender_male'] = (_df['gender'].str.lower() == 'male').astype(int)

# Unadjusted comparisons
mean_affairs = _df.groupby('children_yes')['affairs_count'].mean()
prop_any = _df.assign(any_affair=_df['affairs_count'] > 0).groupby('children_yes')['any_affair'].mean()

print('Mean affairs count by children (0=no,1=yes):')
print(mean_affairs)
print('Proportion any affair by children (0=no,1=yes):')
print(prop_any)

# t-test like via OLS
ols_unadj = smf.ols('affairs_count ~ children_yes', data=_df).fit()
print('\nOLS unadjusted:')
print(ols_unadj.summary())

# Logistic regression for any affair
_df['any_affair'] = (_df['affairs_count'] > 0).astype(int)
logit_unadj = smf.logit('any_affair ~ children_yes', data=_df).fit(disp=False)
print('\nLogit unadjusted:')
print(logit_unadj.summary())

# Adjusted model with covariates
# Exclude the weird 'education' column, not mapped.
ols_adj = smf.ols(
    'affairs_count ~ children_yes + years_married + age_years + education_years + occupation_code + religiousness_scale + marriage_rating + gender_male',
    data=_df
).fit()
print('\nOLS adjusted:')
print(ols_adj.summary())

logit_adj = smf.logit(
    'any_affair ~ children_yes + years_married + age_years + education_years + occupation_code + religiousness_scale + marriage_rating + gender_male',
    data=_df
).fit(disp=False)
print('\nLogit adjusted:')
print(logit_adj.summary())
