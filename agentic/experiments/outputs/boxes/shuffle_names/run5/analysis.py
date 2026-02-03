import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('boxes.csv')

# Map outcome
# 1 = unchosen option (no social info)
# 2 = majority option
# 3 = minority option

# Social reliance: chose demonstrated (majority or minority) vs unchosen
_df['social_reliance'] = _df['majority_first'].isin([2,3]).astype(int)

# Majority preference: among demonstrated choices, chose majority vs minority
_df['majority_choice'] = np.where(_df['majority_first'].isin([2,3]),
                                 (_df['majority_first'] == 2).astype(int),
                                 np.nan)

# Basic summaries by age and site (culture)
summary_by_site = _df.groupby('y').agg(
    n=('majority_first','size'),
    social_reliance_rate=('social_reliance','mean'),
    majority_rate=('majority_first', lambda s: (s==2).mean())
).reset_index()

# Age bands for descriptive view
age_bins = [4,6,8,10,12,14]
_df['age_band'] = pd.cut(_df['age'], bins=age_bins, right=True, include_lowest=True)
summary_by_age = _df.groupby('age_band').agg(
    n=('majority_first','size'),
    social_reliance_rate=('social_reliance','mean'),
    majority_rate=('majority_first', lambda s: (s==2).mean())
).reset_index()

# Logistic regression: social reliance ~ age + site + order (culture)
# Use C(y) for site/culture; include culture (order) as control
social_model = smf.logit('social_reliance ~ age + C(y) + culture', data=_df).fit(disp=False)

# Logistic regression: majority preference among demonstrated choices
_df_demo = _df[_df['majority_first'].isin([2,3])].copy()
majority_model = smf.logit('majority_choice ~ age + C(y) + culture', data=_df_demo).fit(disp=False)

# Likelihood ratio test for site effect (culture differences) by comparing to model without site
social_model_nosite = smf.logit('social_reliance ~ age + culture', data=_df).fit(disp=False)
majority_model_nosite = smf.logit('majority_choice ~ age + culture', data=_df_demo).fit(disp=False)

social_lr = social_model.llr  # vs null with intercept only
social_lr_p = social_model.llr_pvalue

# Explicit LR test for site (C(y))
from scipy import stats
social_lr_site = 2*(social_model.llf - social_model_nosite.llf)
social_df_site = social_model.df_model - social_model_nosite.df_model
social_p_site = stats.chi2.sf(social_lr_site, social_df_site)

majority_lr_site = 2*(majority_model.llf - majority_model_nosite.llf)
majority_df_site = majority_model.df_model - majority_model_nosite.df_model
majority_p_site = stats.chi2.sf(majority_lr_site, majority_df_site)

# Age effects
social_age_coef = social_model.params['age']
social_age_p = social_model.pvalues['age']

majority_age_coef = majority_model.params['age']
majority_age_p = majority_model.pvalues['age']

# Print summaries
print('Summary by site (y as site/culture):')
print(summary_by_site.to_string(index=False))
print('\nSummary by age band:')
print(summary_by_age.to_string(index=False))

print('\nSocial reliance model (logit): social_reliance ~ age + C(y) + culture')
print(social_model.summary().tables[1])
print('\nMajority preference model (logit): majority_choice ~ age + C(y) + culture')
print(majority_model.summary().tables[1])

print('\nLR test for site (C(y)) effect:')
print(f'social_reliance: chi2={social_lr_site:.3f}, df={social_df_site:.0f}, p={social_p_site:.4f}')
print(f'majority_choice: chi2={majority_lr_site:.3f}, df={majority_df_site:.0f}, p={majority_p_site:.4f}')

print('\nAge effects:')
print(f'social_reliance age coef={social_age_coef:.3f}, p={social_age_p:.4f}')
print(f'majority_choice age coef={majority_age_coef:.3f}, p={majority_age_p:.4f}')
