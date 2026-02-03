import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
DF_PATH = 'boxes.csv'
df = pd.read_csv(DF_PATH)

# Outcomes
# Reliance on social info: choose demonstrated option (majority or minority)
df['social_choice'] = (df['y'] != 1).astype(int)
# Preference for majority: among demonstrated options, choose majority

df_demo = df[df['y'].isin([2, 3])].copy()
df_demo['majority_choice'] = (df_demo['y'] == 2).astype(int)

# Age groups for descriptives (4-6,7-9,10-12,13-14)
bins = [3, 6, 9, 12, 14]
labels = ['4-6', '7-9', '10-12', '13-14']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
df_demo['age_group'] = pd.cut(df_demo['age'], bins=bins, labels=labels)

# Model 1: social reliance
m_social = smf.logit('social_choice ~ age + C(culture)', data=df).fit(disp=False)
# Joint Wald test for culture effects
wald_culture_social = m_social.wald_test_terms().table.loc['C(culture)', 'pvalue']
# Age p-value
p_age_social = m_social.pvalues.get('age', float('nan'))

# Model 2: majority preference among demonstrated
m_majority = smf.logit('majority_choice ~ age + C(culture)', data=df_demo).fit(disp=False)
wald_culture_majority = m_majority.wald_test_terms().table.loc['C(culture)', 'pvalue']
p_age_majority = m_majority.pvalues.get('age', float('nan'))

# Test age-by-culture interaction via likelihood ratio test
m_social_int = smf.logit('social_choice ~ age * C(culture)', data=df).fit(disp=False)
llf0_social = m_social.llf
llf1_social = m_social_int.llf
lr_stat_social = 2 * (llf1_social - llf0_social)
# df for interaction: (n_cultures-1) additional terms
n_cultures = df['culture'].nunique()
df_int = n_cultures - 1
p_lr_social = stats.chi2.sf(lr_stat_social, df_int)

m_majority_int = smf.logit('majority_choice ~ age * C(culture)', data=df_demo).fit(disp=False)
llf0_majority = m_majority.llf
llf1_majority = m_majority_int.llf
lr_stat_majority = 2 * (llf1_majority - llf0_majority)
p_lr_majority = stats.chi2.sf(lr_stat_majority, df_int)

# Descriptive rates by culture and age groups
social_by_culture = df.groupby('culture')['social_choice'].mean()
majority_by_culture = df_demo.groupby('culture')['majority_choice'].mean()

social_by_age = df.groupby('age_group')['social_choice'].mean()
majority_by_age = df_demo.groupby('age_group')['majority_choice'].mean()

# Write a brief results summary to stdout for review
print('Social reliance model: age p=%.4f, culture (joint) p=%.4f, age*culture LR p=%.4f' % (
    p_age_social, wald_culture_social, p_lr_social))
print('Majority preference model: age p=%.4f, culture (joint) p=%.4f, age*culture LR p=%.4f' % (
    p_age_majority, wald_culture_majority, p_lr_majority))

print('\nDescriptives:')
print('Social reliance by culture:')
print(social_by_culture.sort_index())
print('Majority preference by culture (demonstrated only):')
print(majority_by_culture.sort_index())
print('Social reliance by age group:')
print(social_by_age)
print('Majority preference by age group:')
print(majority_by_age)
