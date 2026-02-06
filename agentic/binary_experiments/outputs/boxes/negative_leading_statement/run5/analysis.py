import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('boxes.csv')

# Define outcomes
_df['social'] = (_df['y'] != 1).astype(int)  # reliance on social info
_social_df = _df[_df['y'] != 1].copy()
_social_df['majority'] = (_social_df['y'] == 2).astype(int)  # preference for majority among social choices

# Descriptives
print('Overall choice proportions (y):')
print(_df['y'].value_counts(normalize=True).sort_index())
print('\nOverall social reliance (y!=1):', _df['social'].mean())
print('Overall majority preference among social (y==2):', _social_df['majority'].mean())

# By-culture descriptives
print('\nBy-culture social reliance:')
print(_df.groupby('culture')['social'].mean())
print('\nBy-culture majority preference among social:')
print(_social_df.groupby('culture')['majority'].mean())

# By-age descriptives (grouped for stability)
_df['age_group'] = pd.cut(_df['age'], bins=[3,6,9,12,15], labels=['4-6','7-9','10-12','13-14'])
_social_df['age_group'] = _df.loc[_social_df.index, 'age_group']
print('\nBy-age-group social reliance:')
print(_df.groupby('age_group')['social'].mean())
print('\nBy-age-group majority preference among social:')
print(_social_df.groupby('age_group')['majority'].mean())

# Logistic models for inference
model_social = smf.logit('social ~ age + C(culture)', data=_df).fit(disp=0)
model_majority = smf.logit('majority ~ age + C(culture)', data=_social_df).fit(disp=0)

# Likelihood ratio tests for culture and age terms
model_social_no_culture = smf.logit('social ~ age', data=_df).fit(disp=0)
model_majority_no_culture = smf.logit('majority ~ age', data=_social_df).fit(disp=0)

lr_social_culture = 2 * (model_social.llf - model_social_no_culture.llf)
_df_social_culture = model_social.df_model - model_social_no_culture.df_model
p_social_culture = stats.chi2.sf(lr_social_culture, _df_social_culture)

lr_majority_culture = 2 * (model_majority.llf - model_majority_no_culture.llf)
_df_majority_culture = model_majority.df_model - model_majority_no_culture.df_model
p_majority_culture = stats.chi2.sf(lr_majority_culture, _df_majority_culture)

model_social_no_age = smf.logit('social ~ C(culture)', data=_df).fit(disp=0)
model_majority_no_age = smf.logit('majority ~ C(culture)', data=_social_df).fit(disp=0)

lr_social_age = 2 * (model_social.llf - model_social_no_age.llf)
p_social_age = stats.chi2.sf(lr_social_age, 1)

lr_majority_age = 2 * (model_majority.llf - model_majority_no_age.llf)
p_majority_age = stats.chi2.sf(lr_majority_age, 1)

print('\nLogit social ~ age + culture:')
print(model_social.summary())
print('\nLogit majority ~ age + culture (among social choices):')
print(model_majority.summary())

print('\nLR test culture effect on social reliance:', lr_social_culture, 'df', _df_social_culture, 'p', p_social_culture)
print('LR test culture effect on majority preference:', lr_majority_culture, 'df', _df_majority_culture, 'p', p_majority_culture)
print('LR test age effect on social reliance:', lr_social_age, 'p', p_social_age)
print('LR test age effect on majority preference:', lr_majority_age, 'p', p_majority_age)
