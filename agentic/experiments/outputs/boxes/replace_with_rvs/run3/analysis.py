import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def lr_test(full, reduced):
    lr = 2 * (full.llf - reduced.llf)
    df = full.df_model - reduced.df_model
    p = chi2.sf(lr, df)
    return lr, df, p


df = pd.read_csv('boxes.csv')

# Outcome 1: reliance on social information (chose demonstrated option)
df['social'] = (df['y'] != 1).astype(int)

# Outcome 2: preference for majority cues among those who chose a demonstrated option
_demo = df[df['y'] != 1].copy()
_demo['majority'] = (_demo['y'] == 2).astype(int)

results = {}

# Social reliance models
m_social_age = smf.glm('social ~ age', data=df, family=sm.families.Binomial()).fit()
m_social_cult = smf.glm('social ~ C(culture)', data=df, family=sm.families.Binomial()).fit()
m_social_age_cult = smf.glm('social ~ age + C(culture)', data=df, family=sm.families.Binomial()).fit()
m_social_inter = smf.glm('social ~ age * C(culture)', data=df, family=sm.families.Binomial()).fit()

results['social_age_effect'] = lr_test(m_social_age_cult, m_social_cult)
results['social_culture_effect'] = lr_test(m_social_age_cult, m_social_age)
results['social_interaction'] = lr_test(m_social_inter, m_social_age_cult)

# Majority preference models (among demonstrated)
m_maj_age = smf.glm('majority ~ age', data=_demo, family=sm.families.Binomial()).fit()
m_maj_cult = smf.glm('majority ~ C(culture)', data=_demo, family=sm.families.Binomial()).fit()
m_maj_age_cult = smf.glm('majority ~ age + C(culture)', data=_demo, family=sm.families.Binomial()).fit()
m_maj_inter = smf.glm('majority ~ age * C(culture)', data=_demo, family=sm.families.Binomial()).fit()

results['majority_age_effect'] = lr_test(m_maj_age_cult, m_maj_cult)
results['majority_culture_effect'] = lr_test(m_maj_age_cult, m_maj_age)
results['majority_interaction'] = lr_test(m_maj_inter, m_maj_age_cult)

# Directional age effects from additive models
age_coef_social = m_social_age_cult.params.get('age', np.nan)
age_coef_majority = m_maj_age_cult.params.get('age', np.nan)

print('Sample size:', len(df))
print('Demonstrated-choice sample:', len(_demo))
print('\nLikelihood ratio tests (LR, df, p):')
for k, v in results.items():
    print(f'{k}: LR={v[0]:.3f}, df={int(v[1])}, p={v[2]:.4f}')

print('\nAge coefficients (additive models):')
print(f'social reliance age coef: {age_coef_social:.4f}')
print(f'majority preference age coef: {age_coef_majority:.4f}')
