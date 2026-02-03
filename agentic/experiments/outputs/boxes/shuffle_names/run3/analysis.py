import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'boxes.csv'
df = pd.read_csv(path)

# Outcome coding
# 1 = unchosen option, 2 = majority option, 3 = minority option
# Social reliance: chose a demonstrated option (majority or minority)
df['social_reliance'] = df['majority_first'].isin([2, 3]).astype(int)

# Majority preference among demonstrated options only
shown = df[df['majority_first'].isin([2, 3])].copy()
shown['majority_choice'] = (shown['majority_first'] == 2).astype(int)

# Age groups for descriptive summaries
bins = [3, 6, 9, 12, 14]
labels = ['4-6', '7-9', '10-12', '13-14']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
shown['age_group'] = pd.cut(shown['age'], bins=bins, labels=labels)

# Descriptive: proportions by site and age group
social_by_site_age = df.pivot_table(index='y', columns='age_group', values='social_reliance', aggfunc='mean')
majority_by_site_age = shown.pivot_table(index='y', columns='age_group', values='majority_choice', aggfunc='mean')

# Logistic regression: social reliance ~ age + site
social_model = smf.logit('social_reliance ~ age + C(y)', data=df).fit(disp=False)

# Logistic regression: majority preference ~ age + site (among demonstrated)
majority_model = smf.logit('majority_choice ~ age + C(y)', data=shown).fit(disp=False)

# Interaction models to test developmental changes differing by site
social_int_model = smf.logit('social_reliance ~ age * C(y)', data=df).fit(disp=False)
majority_int_model = smf.logit('majority_choice ~ age * C(y)', data=shown).fit(disp=False)

# Summaries
print('Rows:', len(df))
print('Social reliance overall:', df['social_reliance'].mean())
print('Majority preference overall (among demonstrated):', shown['majority_choice'].mean())
print('\nSocial reliance by site and age group:\n', social_by_site_age)
print('\nMajority preference by site and age group:\n', majority_by_site_age)

print('\nSocial reliance model (age + site)\n', social_model.summary())
print('\nMajority preference model (age + site)\n', majority_model.summary())

print('\nSocial reliance interaction model (age * site)\n', social_int_model.summary())
print('\nMajority preference interaction model (age * site)\n', majority_int_model.summary())
