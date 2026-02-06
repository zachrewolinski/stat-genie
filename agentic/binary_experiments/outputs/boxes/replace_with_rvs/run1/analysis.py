import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

# Load data
_df = pd.read_csv('boxes.csv')

# Define outcomes
_df['social_reliance'] = (_df['y'] != 1).astype(int)  # chose demonstrated option
_demo = _df[_df['y'] != 1].copy()
_demo['majority_pref'] = (_demo['y'] == 2).astype(int)  # chose majority among demonstrated

# Logistic regression models
# Social reliance
_sr_main = smf.glm('social_reliance ~ age + C(culture)', data=_df, family=sm.families.Binomial()).fit()
_sr_int = smf.glm('social_reliance ~ age * C(culture)', data=_df, family=sm.families.Binomial()).fit()

# Majority preference (conditional on choosing a demonstrated option)
_mp_main = smf.glm('majority_pref ~ age + C(culture)', data=_demo, family=sm.families.Binomial()).fit()
_mp_int = smf.glm('majority_pref ~ age * C(culture)', data=_demo, family=sm.families.Binomial()).fit()

# Likelihood ratio tests for interaction (age-by-culture)
_sr_lrt = 2 * (_sr_int.llf - _sr_main.llf)
_sr_p_int = chi2.sf(_sr_lrt, _sr_int.df_model - _sr_main.df_model)

_mp_lrt = 2 * (_mp_int.llf - _mp_main.llf)
_mp_p_int = chi2.sf(_mp_lrt, _mp_int.df_model - _mp_main.df_model)

# Main effects tests (culture and age)
_sr_age_only = smf.glm('social_reliance ~ age', data=_df, family=sm.families.Binomial()).fit()
_sr_culture_p = chi2.sf(2 * (_sr_main.llf - _sr_age_only.llf), _sr_main.df_model - _sr_age_only.df_model)

_mp_age_only = smf.glm('majority_pref ~ age', data=_demo, family=sm.families.Binomial()).fit()
_mp_culture_p = chi2.sf(2 * (_mp_main.llf - _mp_age_only.llf), _mp_main.df_model - _mp_age_only.df_model)

_sr_culture_only = smf.glm('social_reliance ~ C(culture)', data=_df, family=sm.families.Binomial()).fit()
_sr_age_p = chi2.sf(2 * (_sr_main.llf - _sr_culture_only.llf), _sr_main.df_model - _sr_culture_only.df_model)

_mp_culture_only = smf.glm('majority_pref ~ C(culture)', data=_demo, family=sm.families.Binomial()).fit()
_mp_age_p = chi2.sf(2 * (_mp_main.llf - _mp_culture_only.llf), _mp_main.df_model - _mp_culture_only.df_model)

# Descriptive summaries
_bins = [3.5, 7.5, 10.5, 14.5]
_labels = ['4-7', '8-10', '11-14']

_df['age_group'] = pd.cut(_df['age'], bins=_bins, labels=_labels)
_demo['age_group'] = pd.cut(_demo['age'], bins=_bins, labels=_labels)

_sr_by_age = _df.groupby('age_group')['social_reliance'].mean()
_mp_by_age = _demo.groupby('age_group')['majority_pref'].mean()

_sr_by_culture = _df.groupby('culture')['social_reliance'].mean()
_mp_by_culture = _demo.groupby('culture')['majority_pref'].mean()

print('Social reliance: culture main effect p =', _sr_culture_p)
print('Social reliance: age main effect p =', _sr_age_p)
print('Social reliance: age*culture interaction p =', _sr_p_int)
print('Majority preference: culture main effect p =', _mp_culture_p)
print('Majority preference: age main effect p =', _mp_age_p)
print('Majority preference: age*culture interaction p =', _mp_p_int)

print('\nSocial reliance by age group')
print(_sr_by_age)
print('\nMajority preference by age group')
print(_mp_by_age)
print('\nSocial reliance by culture')
print(_sr_by_culture)
print('\nMajority preference by culture')
print(_mp_by_culture)
