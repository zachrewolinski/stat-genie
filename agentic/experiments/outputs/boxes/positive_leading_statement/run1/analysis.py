import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'boxes.csv'
df = pd.read_csv(path)

# Derived outcomes
# Social reliance: choosing demonstrated option (majority or minority)
df['social'] = (df['y'] != 1).astype(int)
# Majority preference among social choices
social_df = df[df['y'].isin([2, 3])].copy()
social_df['majority_choice'] = (social_df['y'] == 2).astype(int)

# Helper: likelihood ratio test between nested models

def lr_test(full_result, reduced_result, df_diff):
    lr_stat = 2 * (full_result.llf - reduced_result.llf)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, p_value

# Model 1: Social reliance ~ age + culture
model_social_full = smf.glm('social ~ age + C(culture)', data=df, family=sm.families.Binomial()).fit()
model_social_age = smf.glm('social ~ age', data=df, family=sm.families.Binomial()).fit()
model_social_culture = smf.glm('social ~ C(culture)', data=df, family=sm.families.Binomial()).fit()

# Model 2: Majority preference among social choices ~ age + culture
model_major_full = smf.glm('majority_choice ~ age + C(culture)', data=social_df, family=sm.families.Binomial()).fit()
model_major_age = smf.glm('majority_choice ~ age', data=social_df, family=sm.families.Binomial()).fit()
model_major_culture = smf.glm('majority_choice ~ C(culture)', data=social_df, family=sm.families.Binomial()).fit()

# Likelihood ratio tests for culture effect controlling for age
lr_social_culture = lr_test(model_social_full, model_social_age, df_diff=model_social_full.df_model - model_social_age.df_model)
lr_major_culture = lr_test(model_major_full, model_major_age, df_diff=model_major_full.df_model - model_major_age.df_model)

# Likelihood ratio tests for age effect controlling for culture
lr_social_age = lr_test(model_social_full, model_social_culture, df_diff=model_social_full.df_model - model_social_culture.df_model)
lr_major_age = lr_test(model_major_full, model_major_culture, df_diff=model_major_full.df_model - model_major_culture.df_model)

# Descriptive: overall proportions
social_rate = df['social'].mean()
major_rate = social_df['majority_choice'].mean()

# Descriptive by culture and age group
age_bins = [4, 7, 10, 13, 15]
age_labels = ['4-6', '7-9', '10-12', '13-14']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
social_df['age_group'] = pd.cut(social_df['age'], bins=age_bins, labels=age_labels, right=False)

social_by_culture = df.groupby('culture')['social'].mean()
major_by_culture = social_df.groupby('culture')['majority_choice'].mean()

social_by_age = df.groupby('age_group')['social'].mean()
major_by_age = social_df.groupby('age_group')['majority_choice'].mean()

print('Overall social reliance rate:', round(social_rate, 3))
print('Overall majority choice rate (among social choices):', round(major_rate, 3))
print('\nSocial reliance by culture (mean):')
print(social_by_culture.round(3))
print('\nMajority preference by culture (mean):')
print(major_by_culture.round(3))
print('\nSocial reliance by age group (mean):')
print(social_by_age.round(3))
print('\nMajority preference by age group (mean):')
print(major_by_age.round(3))

print('\nModel tests:')
print('Social reliance: culture effect LR stat {:.3f}, p={:.4g}'.format(*lr_social_culture))
print('Social reliance: age effect LR stat {:.3f}, p={:.4g}'.format(*lr_social_age))
print('Majority preference: culture effect LR stat {:.3f}, p={:.4g}'.format(*lr_major_culture))
print('Majority preference: age effect LR stat {:.3f}, p={:.4g}'.format(*lr_major_age))

# Also print age coefficient direction for interpretation
print('\nSocial reliance age coef:', model_social_full.params.get('age', np.nan))
print('Majority preference age coef:', model_major_full.params.get('age', np.nan))
