import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
Df = pd.read_csv('boxes.csv')

# Outcomes
Df['choose_social'] = Df['majority_first'].isin([2, 3]).astype(int)  # any demonstrated option
Df['choose_majority'] = (Df['majority_first'] == 2).astype(int)      # majority option

# Treat site/culture id as categorical
Df['y'] = Df['y'].astype(int).astype('category')

# Helper: likelihood-ratio test

def lr_test(full, reduced):
    lr = 2 * (full.llf - reduced.llf)
    df = full.df_model - reduced.df_model
    p = stats.chi2.sf(lr, df)
    return lr, df, p

# Models for reliance on social information
m_social = smf.logit('choose_social ~ age + C(y)', data=Df).fit(disp=False)
m_social_no_age = smf.logit('choose_social ~ C(y)', data=Df).fit(disp=False)
m_social_no_y = smf.logit('choose_social ~ age', data=Df).fit(disp=False)

# Models for majority preference
m_majority = smf.logit('choose_majority ~ age + C(y)', data=Df).fit(disp=False)
m_majority_no_age = smf.logit('choose_majority ~ C(y)', data=Df).fit(disp=False)
m_majority_no_y = smf.logit('choose_majority ~ age', data=Df).fit(disp=False)

# Interaction tests (age by site)
m_social_inter = smf.logit('choose_social ~ age * C(y)', data=Df).fit(disp=False)
m_majority_inter = smf.logit('choose_majority ~ age * C(y)', data=Df).fit(disp=False)

# Summaries
print('Overall reliance on social information:', Df['choose_social'].mean())
print('Overall majority preference:', Df['choose_majority'].mean())

print('Social reliance: age effect LRT', lr_test(m_social, m_social_no_age))
print('Social reliance: site effect LRT', lr_test(m_social, m_social_no_y))
print('Social reliance: age*site interaction LRT', lr_test(m_social_inter, m_social))

print('Majority preference: age effect LRT', lr_test(m_majority, m_majority_no_age))
print('Majority preference: site effect LRT', lr_test(m_majority, m_majority_no_y))
print('Majority preference: age*site interaction LRT', lr_test(m_majority_inter, m_majority))

# Descriptive rates by age group and site
Df['age_group'] = pd.cut(
    Df['age'],
    bins=[3.99, 5.99, 7.99, 9.99, 11.99, 13.99, 14.99],
    labels=['4-5', '6-7', '8-9', '10-11', '12-13', '14'],
    right=True,
)
summary = Df.groupby(['y', 'age_group'])[['choose_social', 'choose_majority']].mean()
print('\nRates by site and age group (proportions):')
print(summary)
