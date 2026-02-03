import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('boxes.csv')

# Social information reliance: choosing a demonstrated option (majority or minority)
_df['social_info'] = (_df['y'] != 1).astype(int)

# Models for social information reliance
m_social_full = smf.glm('social_info ~ age + C(culture)', data=_df, family=sm.families.Binomial()).fit()
m_social_age = smf.glm('social_info ~ age', data=_df, family=sm.families.Binomial()).fit()
m_social_cult = smf.glm('social_info ~ C(culture)', data=_df, family=sm.families.Binomial()).fit()

# Likelihood ratio tests for culture and age effects
lr_social_culture = 2 * (m_social_full.llf - m_social_age.llf)
df_social_culture = m_social_full.df_model - m_social_age.df_model
p_social_culture = stats.chi2.sf(lr_social_culture, df_social_culture)

lr_social_age = 2 * (m_social_full.llf - m_social_cult.llf)
df_social_age = m_social_full.df_model - m_social_cult.df_model
p_social_age = stats.chi2.sf(lr_social_age, df_social_age)

# Majority preference among demonstrated options only
_demo = _df[_df['y'].isin([2, 3])].copy()
_demo['majority_choice'] = (_demo['y'] == 2).astype(int)

# Models for majority preference
m_maj_full = smf.glm('majority_choice ~ age + C(culture)', data=_demo, family=sm.families.Binomial()).fit()
m_maj_age = smf.glm('majority_choice ~ age', data=_demo, family=sm.families.Binomial()).fit()
m_maj_cult = smf.glm('majority_choice ~ C(culture)', data=_demo, family=sm.families.Binomial()).fit()

# Likelihood ratio tests for culture and age effects
lr_maj_culture = 2 * (m_maj_full.llf - m_maj_age.llf)
df_maj_culture = m_maj_full.df_model - m_maj_age.df_model
p_maj_culture = stats.chi2.sf(lr_maj_culture, df_maj_culture)

lr_maj_age = 2 * (m_maj_full.llf - m_maj_cult.llf)
df_maj_age = m_maj_full.df_model - m_maj_cult.df_model
p_maj_age = stats.chi2.sf(lr_maj_age, df_maj_age)

print('Social information reliance (choose demonstrated option vs unchosen):')
print(m_social_full.summary())
print('\nLRT culture effect (social_info):', lr_social_culture, 'df=', df_social_culture, 'p=', p_social_culture)
print('LRT age effect (social_info):', lr_social_age, 'df=', df_social_age, 'p=', p_social_age)

print('\nMajority preference among demonstrated options (majority vs minority):')
print(m_maj_full.summary())
print('\nLRT culture effect (majority_choice):', lr_maj_culture, 'df=', df_maj_culture, 'p=', p_maj_culture)
print('LRT age effect (majority_choice):', lr_maj_age, 'df=', df_maj_age, 'p=', p_maj_age)
