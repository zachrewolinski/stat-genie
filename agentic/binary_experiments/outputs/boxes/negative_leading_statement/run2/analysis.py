import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('boxes.csv')

# Derived variables
_df['social'] = (_df['y'] != 1).astype(int)  # reliance on social information

# Majority preference among those choosing demonstrated options
_demo = _df[_df['y'].isin([2, 3])].copy()
_demo['majority'] = (_demo['y'] == 2).astype(int)

# Descriptive summaries
print('Counts by outcome (y):')
print(_df['y'].value_counts().sort_index())
print('\nMean social reliance by culture:')
print(_df.groupby('culture')['social'].mean())
print('\nMean majority choice (among demonstrated) by culture:')
print(_demo.groupby('culture')['majority'].mean())

# Logistic regression: social reliance ~ age * culture
m_social_full = smf.logit('social ~ age * C(culture)', data=_df).fit(disp=False)
m_social_no_int = smf.logit('social ~ age + C(culture)', data=_df).fit(disp=False)
m_social_no_culture = smf.logit('social ~ age', data=_df).fit(disp=False)
m_social_no_age = smf.logit('social ~ C(culture)', data=_df).fit(disp=False)

lr_int = 2 * (m_social_full.llf - m_social_no_int.llf)
df_int = m_social_full.df_model - m_social_no_int.df_model
p_int = stats.chi2.sf(lr_int, df_int)

lr_cult = 2 * (m_social_no_int.llf - m_social_no_culture.llf)
df_cult = m_social_no_int.df_model - m_social_no_culture.df_model
p_cult = stats.chi2.sf(lr_cult, df_cult)

lr_age = 2 * (m_social_no_int.llf - m_social_no_age.llf)
df_age = m_social_no_int.df_model - m_social_no_age.df_model
p_age = stats.chi2.sf(lr_age, df_age)

print('\nSocial reliance model (logit):')
print(f'Interaction age*culture LR p-value: {p_int:.4g}')
print(f'Culture main effect LR p-value (no int): {p_cult:.4g}')
print(f'Age main effect LR p-value (no int): {p_age:.4g}')

# Logistic regression: majority preference among demonstrated ~ age * culture
m_maj_full = smf.logit('majority ~ age * C(culture)', data=_demo).fit(disp=False)
m_maj_no_int = smf.logit('majority ~ age + C(culture)', data=_demo).fit(disp=False)
m_maj_no_culture = smf.logit('majority ~ age', data=_demo).fit(disp=False)
m_maj_no_age = smf.logit('majority ~ C(culture)', data=_demo).fit(disp=False)

lr2_int = 2 * (m_maj_full.llf - m_maj_no_int.llf)
df2_int = m_maj_full.df_model - m_maj_no_int.df_model
p2_int = stats.chi2.sf(lr2_int, df2_int)

lr2_cult = 2 * (m_maj_no_int.llf - m_maj_no_culture.llf)
df2_cult = m_maj_no_int.df_model - m_maj_no_culture.df_model
p2_cult = stats.chi2.sf(lr2_cult, df2_cult)

lr2_age = 2 * (m_maj_no_int.llf - m_maj_no_age.llf)
df2_age = m_maj_no_int.df_model - m_maj_no_age.df_model
p2_age = stats.chi2.sf(lr2_age, df2_age)

print('\nMajority preference model (logit among demonstrated choices):')
print(f'Interaction age*culture LR p-value: {p2_int:.4g}')
print(f'Culture main effect LR p-value (no int): {p2_cult:.4g}')
print(f'Age main effect LR p-value (no int): {p2_age:.4g}')
