import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2, chi2_contingency

# Load data
DF_PATH = 'boxes.csv'
df = pd.read_csv(DF_PATH)

# Derived variables
# Social reliance: choosing any demonstrated option (majority or minority) vs unchosen
# Majority preference: among demonstrated choices, choosing majority vs minority

df['social_choice'] = df['y'].isin([2, 3]).astype(int)
social = df[df['y'].isin([2, 3])].copy()
social['majority_choice'] = (social['y'] == 2).astype(int)

# Age groups for descriptive tables
bins = [4, 6, 9, 12, 15]
labels = ['4-6', '7-9', '10-12', '13-14']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True)
social['age_group'] = pd.cut(social['age'], bins=bins, labels=labels, right=True, include_lowest=True)

print('Counts')
print('Total N:', len(df))
print('Social choice rate:', df['social_choice'].mean())
print('Majority choice rate (among social choices):', social['majority_choice'].mean())
print()

print('Social choice by culture (proportions)')
print(pd.crosstab(df['culture'], df['social_choice'], normalize='index'))
print()
print('Majority vs minority by culture (proportions)')
print(pd.crosstab(social['culture'], social['y'], normalize='index'))
print()

print('Social choice by age group (proportions)')
print(pd.crosstab(df['age_group'], df['social_choice'], normalize='index'))
print()
print('Majority vs minority by age group (proportions)')
print(pd.crosstab(social['age_group'], social['y'], normalize='index'))
print()

# Chi-square tests for variation across cultures and age groups
ct = pd.crosstab(df['culture'], df['social_choice'])
chi2_val, p_social_culture, _, _ = chi2_contingency(ct)

ct2 = pd.crosstab(df['age_group'], df['social_choice'])
chi2_val2, p_social_age, _, _ = chi2_contingency(ct2)

ct3 = pd.crosstab(social['culture'], social['y'])
chi2_val3, p_majority_culture, _, _ = chi2_contingency(ct3)

ct4 = pd.crosstab(social['age_group'], social['y'])
chi2_val4, p_majority_age, _, _ = chi2_contingency(ct4)

print('Chi-square p-values')
print('Social choice ~ culture:', p_social_culture)
print('Social choice ~ age_group:', p_social_age)
print('Majority vs minority ~ culture:', p_majority_culture)
print('Majority vs minority ~ age_group:', p_majority_age)
print()

# Logistic regression with controls
# 1) Social reliance (social_choice) vs not
full1 = smf.logit('social_choice ~ age + C(culture) + gender + majority_first', data=df).fit(disp=0)
red1 = smf.logit('social_choice ~ age + gender + majority_first', data=df).fit(disp=0)
LR1 = 2 * (full1.llf - red1.llf)
df1 = full1.df_model - red1.df_model
p_culture_social = chi2.sf(LR1, df1)

# 2) Majority preference among social choices
full2 = smf.logit('majority_choice ~ age + C(culture) + gender + majority_first', data=social).fit(disp=0)
red2 = smf.logit('majority_choice ~ age + gender + majority_first', data=social).fit(disp=0)
LR2 = 2 * (full2.llf - red2.llf)
df2 = full2.df_model - red2.df_model
p_culture_majority = chi2.sf(LR2, df2)

print('Logistic regression results')
print('Social choice model: age p-value =', full1.pvalues['age'])
print('Social choice model: culture LR p-value =', p_culture_social)
print('Majority preference model: age p-value =', full2.pvalues['age'])
print('Majority preference model: culture LR p-value =', p_culture_majority)
print()

print('Social choice model summary')
print(full1.summary())
print()
print('Majority preference model summary')
print(full2.summary())
