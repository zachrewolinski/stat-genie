import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('boxes.csv')

# Define outcomes
_df['social'] = (_df['feature1'] != 1).astype(int)  # relied on social info
_df['majority'] = (_df['feature1'] == 2).astype(int)  # chose majority option

# Logistic regression: social reliance ~ age + site
m_social_full = smf.logit('social ~ feature3 + C(feature5)', data=_df).fit(disp=False)
m_social_red = smf.logit('social ~ feature3', data=_df).fit(disp=False)

# Logistic regression: majority preference ~ age + site
m_majority_full = smf.logit('majority ~ feature3 + C(feature5)', data=_df).fit(disp=False)
m_majority_red = smf.logit('majority ~ feature3', data=_df).fit(disp=False)

# Likelihood-ratio tests for overall site effect
lr_social = 2 * (m_social_full.llf - m_social_red.llf)
site_df_social = m_social_full.df_model - m_social_red.df_model
p_social_site = stats.chi2.sf(lr_social, df=site_df_social)

lr_majority = 2 * (m_majority_full.llf - m_majority_red.llf)
site_df_majority = m_majority_full.df_model - m_majority_red.df_model
p_majority_site = stats.chi2.sf(lr_majority, df=site_df_majority)

# Age effects (Wald tests from full models)
p_social_age = float(m_social_full.pvalues['feature3'])
p_majority_age = float(m_majority_full.pvalues['feature3'])

# Descriptives for context
site_social = _df.groupby('feature5')['social'].mean()
site_majority = _df.groupby('feature5')['majority'].mean()

age_bins = [3, 6, 9, 12, 14]
age_labels = ['4-6', '7-9', '10-12', '13-14']
_df['age_group'] = pd.cut(_df['feature3'], bins=age_bins, labels=age_labels)

age_social = _df.groupby('age_group')['social'].mean()
age_majority = _df.groupby('age_group')['majority'].mean()

print('N:', len(_df))
print('Outcome counts:', _df['feature1'].value_counts().sort_index().to_dict())
print('\nLogit: social ~ age + site')
print(m_social_full.summary())
print('LR test site effect p:', p_social_site)
print('Age effect p:', p_social_age)

print('\nLogit: majority ~ age + site')
print(m_majority_full.summary())
print('LR test site effect p:', p_majority_site)
print('Age effect p:', p_majority_age)

print('\nSite-level proportions (social reliance):')
print(site_social)
print('\nSite-level proportions (majority preference):')
print(site_majority)

print('\nAge-group proportions (social reliance):')
print(age_social)
print('\nAge-group proportions (majority preference):')
print(age_majority)
