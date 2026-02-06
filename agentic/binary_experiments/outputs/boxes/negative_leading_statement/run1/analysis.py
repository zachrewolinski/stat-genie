import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('boxes.csv')

# Create outcomes
_df['demo_choice'] = (_df['y'] != 1).astype(int)  # chose a demonstrated option
_df['majority_choice'] = (_df['y'] == 2).astype(int)
_df_demo = _df[_df['y'] != 1].copy()  # only demonstrated choices for majority vs minority
_df_demo['majority_choice'] = (_df_demo['y'] == 2).astype(int)

# Helper for LR test
def lr_test(model_restricted, model_full):
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    df_diff = model_full.df_model - model_restricted.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value

results = {}

# 1) Reliance on social information (demo_choice)
model_null = smf.logit('demo_choice ~ 1', data=_df).fit(disp=False)
model_main = smf.logit('demo_choice ~ age + C(culture)', data=_df).fit(disp=False)
model_int = smf.logit('demo_choice ~ age * C(culture)', data=_df).fit(disp=False)

lr_main = lr_test(model_null, model_main)
lr_int = lr_test(model_main, model_int)

results['demo_choice'] = {
    'n': len(_df),
    'rate': _df['demo_choice'].mean(),
    'lr_main': lr_main,
    'lr_int': lr_int,
    'age_p': model_main.pvalues.get('age', np.nan),
}

# 2) Preference for majority cues (majority vs minority among demonstrated)
model_null_m = smf.logit('majority_choice ~ 1', data=_df_demo).fit(disp=False)
model_main_m = smf.logit('majority_choice ~ age + C(culture)', data=_df_demo).fit(disp=False)
model_int_m = smf.logit('majority_choice ~ age * C(culture)', data=_df_demo).fit(disp=False)

lr_main_m = lr_test(model_null_m, model_main_m)
lr_int_m = lr_test(model_main_m, model_int_m)

results['majority_choice'] = {
    'n': len(_df_demo),
    'rate': _df_demo['majority_choice'].mean(),
    'lr_main': lr_main_m,
    'lr_int': lr_int_m,
    'age_p': model_main_m.pvalues.get('age', np.nan),
}

# Also compute simple chi-square tests for y by culture and y by age band
_df['age_band'] = pd.cut(_df['age'], bins=[3.5, 6.5, 9.5, 12.5, 14.5], labels=['4-6', '7-9', '10-12', '13-14'])

ct_culture = pd.crosstab(_df['culture'], _df['y'])
ct_age = pd.crosstab(_df['age_band'], _df['y'])

chi2_culture = stats.chi2_contingency(ct_culture)
chi2_age = stats.chi2_contingency(ct_age)

results['chi2_culture'] = {'chi2': chi2_culture[0], 'df': chi2_culture[2], 'p': chi2_culture[1]}
results['chi2_age'] = {'chi2': chi2_age[0], 'df': chi2_age[2], 'p': chi2_age[1]}

# Print results for inspection
print('Reliance on social info (demo_choice)')
print(results['demo_choice'])
print('Preference for majority (among demonstrated)')
print(results['majority_choice'])
print('Chi-square y by culture:', results['chi2_culture'])
print('Chi-square y by age band:', results['chi2_age'])
