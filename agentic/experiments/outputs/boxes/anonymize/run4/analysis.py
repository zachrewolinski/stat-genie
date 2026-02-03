import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('boxes.csv')

# Rename for clarity
_df = _df.rename(columns={
    'feature1': 'choice',
    'feature2': 'gender',
    'feature3': 'age',
    'feature4': 'majority_first',
    'feature5': 'site'
})

# Map outcomes
# choice: 1=unchosen, 2=majority, 3=minority
_df['social_choice'] = _df['choice'].isin([2, 3]).astype(int)  # reliance on social info
_df['majority_choice'] = (_df['choice'] == 2).astype(int)      # preference for majority
_df['minority_choice'] = (_df['choice'] == 3).astype(int)

# Age groups for developmental stages
bins = [3, 6, 9, 12, 15]
labels = ['4-6', '7-9', '10-12', '13-14']
_df['age_group'] = pd.cut(_df['age'], bins=bins, labels=labels, right=True)

# Descriptive stats by site and age group
site_summary = _df.groupby('site')[['social_choice', 'majority_choice', 'minority_choice']].mean()
age_summary = _df.groupby('age_group')[['social_choice', 'majority_choice', 'minority_choice']].mean()

# Logistic regression: social reliance ~ age + site
social_model = smf.glm('social_choice ~ age + C(site)', data=_df, family=sm.families.Binomial()).fit()

# Logistic regression: majority preference among all choices ~ age + site
majority_model = smf.glm('majority_choice ~ age + C(site)', data=_df, family=sm.families.Binomial()).fit()

# Logistic regression: majority vs minority among social choices only
social_only = _df[_df['social_choice'] == 1].copy()
majority_vs_minority = smf.glm('majority_choice ~ age + C(site)', data=social_only, family=sm.families.Binomial()).fit()

# Prepare results
print('Rows:', len(_df))
print('\nOverall choice proportions:')
print(_df['choice'].value_counts(normalize=True).sort_index())

print('\nSite-level means:')
print(site_summary)

print('\nAge-group means:')
print(age_summary)

print('\nSocial reliance model:')
print(social_model.summary())

print('\nMajority preference model (all choices):')
print(majority_model.summary())

print('\nMajority vs minority among social choices:')
print(majority_vs_minority.summary())
