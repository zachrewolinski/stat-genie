import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('boxes.csv')

# Outcome 1: reliance on social information (choose demonstrated option)
_df['social'] = _df['y'].isin([2, 3]).astype(int)

# Outcome 2: majority preference among those who chose demonstrated options
_demo = _df[_df['y'].isin([2, 3])].copy()
_demo['majority'] = (_demo['y'] == 2).astype(int)

# Helper: likelihood-ratio test

def lr_test(llf_full, llf_reduced, df_diff):
    lr = 2 * (llf_full - llf_reduced)
    p = stats.chi2.sf(lr, df_diff)
    return lr, p

# Model for social reliance
formula_full = 'social ~ age + C(culture) + gender + majority_first'
model_full = smf.glm(formula_full, data=_df, family=sm.families.Binomial()).fit()

# Reduced models to test age and culture effects
model_no_age = smf.glm('social ~ C(culture) + gender + majority_first', data=_df, family=sm.families.Binomial()).fit()
model_no_culture = smf.glm('social ~ age + gender + majority_first', data=_df, family=sm.families.Binomial()).fit()

lr_age_social, p_age_social_lr = lr_test(model_full.llf, model_no_age.llf, model_full.df_model - model_no_age.df_model)
lr_cult_social, p_cult_social_lr = lr_test(model_full.llf, model_no_culture.llf, model_full.df_model - model_no_culture.df_model)

# Model for majority preference
formula_full_majority = 'majority ~ age + C(culture) + gender + majority_first'
model_full_majority = smf.glm(formula_full_majority, data=_demo, family=sm.families.Binomial()).fit()

model_no_age_majority = smf.glm('majority ~ C(culture) + gender + majority_first', data=_demo, family=sm.families.Binomial()).fit()
model_no_culture_majority = smf.glm('majority ~ age + gender + majority_first', data=_demo, family=sm.families.Binomial()).fit()

lr_age_maj, p_age_maj_lr = lr_test(model_full_majority.llf, model_no_age_majority.llf, model_full_majority.df_model - model_no_age_majority.df_model)
lr_cult_maj, p_cult_maj_lr = lr_test(model_full_majority.llf, model_no_culture_majority.llf, model_full_majority.df_model - model_no_culture_majority.df_model)

# Descriptive summaries
social_by_culture = _df.groupby('culture')['social'].mean()
maj_by_culture = _demo.groupby('culture')['majority'].mean()

# Age bins for descriptive trends
bins = [3, 6, 9, 12, 14]
labels = ['4-6', '7-9', '10-12', '13-14']
_df['age_group'] = pd.cut(_df['age'], bins=bins, labels=labels, include_lowest=True)
_demo['age_group'] = pd.cut(_demo['age'], bins=bins, labels=labels, include_lowest=True)

social_by_age = _df.groupby('age_group')['social'].mean()
maj_by_age = _demo.groupby('age_group')['majority'].mean()

# Chi-square tests for association (non-parametric)
def chi_square(table):
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return chi2, p, dof

chi2_culture_y, p_culture_y_chi, dof_culture_y = chi_square(pd.crosstab(_df['culture'], _df['y']))
chi2_age_y, p_age_y_chi, dof_age_y = chi_square(pd.crosstab(_df['age_group'], _df['y']))
chi2_culture_social, p_culture_social_chi, dof_culture_social = chi_square(pd.crosstab(_df['culture'], _df['social']))
chi2_age_social, p_age_social_chi, dof_age_social = chi_square(pd.crosstab(_df['age_group'], _df['social']))
chi2_culture_maj, p_culture_maj_chi, dof_culture_maj = chi_square(pd.crosstab(_demo['culture'], _demo['majority']))
chi2_age_maj, p_age_maj_chi, dof_age_maj = chi_square(pd.crosstab(_demo['age_group'], _demo['majority']))

# Write concise analysis summary
with open('analysis_summary.txt', 'w') as f:
    f.write('Social reliance model (social ~ age + culture + controls):\n')
    f.write(f'  LR test age: chi2={lr_age_social:.3f}, p={p_age_social_lr:.4g}\n')
    f.write(f'  LR test culture: chi2={lr_cult_social:.3f}, p={p_cult_social_lr:.4g}\n')
    f.write('\nMajority preference model (majority ~ age + culture + controls):\n')
    f.write(f'  LR test age: chi2={lr_age_maj:.3f}, p={p_age_maj_lr:.4g}\n')
    f.write(f'  LR test culture: chi2={lr_cult_maj:.3f}, p={p_cult_maj_lr:.4g}\n')
    f.write('\nDescriptive rates by culture:\n')
    f.write('  Social reliance (choose demonstrated option):\n')
    f.write(social_by_culture.to_string() + '\n')
    f.write('  Majority preference (among demonstrated choices):\n')
    f.write(maj_by_culture.to_string() + '\n')
    f.write('\nDescriptive rates by age group:\n')
    f.write('  Social reliance (choose demonstrated option):\n')
    f.write(social_by_age.to_string() + '\n')
    f.write('  Majority preference (among demonstrated choices):\n')
    f.write(maj_by_age.to_string() + '\n')
    f.write('\nChi-square tests (association):\n')
    f.write(f'  culture vs y: chi2={chi2_culture_y:.3f}, dof={dof_culture_y}, p={p_culture_y_chi:.4g}\n')
    f.write(f'  age_group vs y: chi2={chi2_age_y:.3f}, dof={dof_age_y}, p={p_age_y_chi:.4g}\n')
    f.write(f'  culture vs social: chi2={chi2_culture_social:.3f}, dof={dof_culture_social}, p={p_culture_social_chi:.4g}\n')
    f.write(f'  age_group vs social: chi2={chi2_age_social:.3f}, dof={dof_age_social}, p={p_age_social_chi:.4g}\n')
    f.write(f'  culture vs majority: chi2={chi2_culture_maj:.3f}, dof={dof_culture_maj}, p={p_culture_maj_chi:.4g}\n')
    f.write(f'  age_group vs majority: chi2={chi2_age_maj:.3f}, dof={dof_age_maj}, p={p_age_maj_chi:.4g}\n')

# Also print key results to stdout for quick inspection
print('Social reliance model:')
print(f'  LR test age: chi2={lr_age_social:.3f}, p={p_age_social_lr:.4g}')
print(f'  LR test culture: chi2={lr_cult_social:.3f}, p={p_cult_social_lr:.4g}')
print('Majority preference model:')
print(f'  LR test age: chi2={lr_age_maj:.3f}, p={p_age_maj_lr:.4g}')
print(f'  LR test culture: chi2={lr_cult_maj:.3f}, p={p_cult_maj_lr:.4g}')
print('Chi-square tests:')
print(f'  culture vs y: chi2={chi2_culture_y:.3f}, p={p_culture_y_chi:.4g}')
print(f'  age_group vs y: chi2={chi2_age_y:.3f}, p={p_age_y_chi:.4g}')
print(f'  culture vs social: chi2={chi2_culture_social:.3f}, p={p_culture_social_chi:.4g}')
print(f'  age_group vs social: chi2={chi2_age_social:.3f}, p={p_age_social_chi:.4g}')
print(f'  culture vs majority: chi2={chi2_culture_maj:.3f}, p={p_culture_maj_chi:.4g}')
print(f'  age_group vs majority: chi2={chi2_age_maj:.3f}, p={p_age_maj_chi:.4g}')
