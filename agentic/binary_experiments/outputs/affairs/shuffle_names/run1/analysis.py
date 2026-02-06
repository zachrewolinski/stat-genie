import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Map shuffled columns to their actual meanings based on value patterns
# affairs count (0,1,2,3,7,12) is in the 'age' column
_df['affairs_count'] = _df['age']
# children indicator (yes/no) is in 'religiousness'
_df['has_children'] = _df['religiousness'].map({'yes': 1, 'no': 0})
# other covariates (optional controls)
_df['age_cat'] = _df['occupation']
_df['years_married'] = _df['children']
_df['religiousness_score'] = _df['rating']
_df['education_level'] = _df['yearsmarried']
_df['occupation_score'] = _df['rownames']
_df['marriage_rating'] = _df['affairs']

# Basic comparisons
summary = (
    _df.groupby('has_children')['affairs_count']
    .agg(['count', 'mean', 'median'])
)
summary['any_affair_rate'] = (
    _df.groupby('has_children')['affairs_count']
    .apply(lambda s: (s > 0).mean())
)

print('Affairs by children (0=no, 1=yes):')
print(summary)

# Difference in means
mean_no = summary.loc[0, 'mean']
mean_yes = summary.loc[1, 'mean']
print('\nMean difference (has_children=1 minus 0):', mean_yes - mean_no)

# Logistic regression: any affair vs. children (with and without controls)
_df['any_affair'] = (_df['affairs_count'] > 0).astype(int)

# Unadjusted model
logit_unadj = smf.logit('any_affair ~ has_children', data=_df).fit(disp=False)
print('\nUnadjusted logit:')
print(logit_unadj.summary())

# Adjusted model with common controls
logit_adj = smf.logit(
    'any_affair ~ has_children + age_cat + years_married + religiousness_score + '
    'education_level + occupation_score + marriage_rating + C(gender)',
    data=_df
).fit(disp=False)
print('\nAdjusted logit:')
print(logit_adj.summary())

# Odds ratios for has_children
or_unadj = np.exp(logit_unadj.params['has_children'])
or_adj = np.exp(logit_adj.params['has_children'])
print('\nOdds ratio (has_children): unadjusted', or_unadj)
print('Odds ratio (has_children): adjusted', or_adj)
