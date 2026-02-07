import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2

# Load data
_df = pd.read_csv('boxes.csv')

# Rename for clarity
_df = _df.rename(columns={
    'feature1': 'outcome',
    'feature2': 'gender',
    'feature3': 'age',
    'feature4': 'majority_first',
    'feature5': 'site'
})

# Outcome coding: 2 = majority option
_df['majority_choice'] = (_df['outcome'] == 2).astype(int)

# Basic summaries
overall_rate = _df['majority_choice'].mean()
site_rates = _df.groupby('site')['majority_choice'].mean()
age_rates = _df.groupby('age')['majority_choice'].mean()

# Logistic regression: main effects
model_main = smf.logit('majority_choice ~ age + C(site)', data=_df).fit(disp=0)

# Logistic regression: add age*site interaction to test differential age trends by site
model_inter = smf.logit('majority_choice ~ age * C(site)', data=_df).fit(disp=0)

# Likelihood ratio test for interaction
lr_stat = 2 * (model_inter.llf - model_main.llf)
lr_df = model_inter.df_model - model_main.df_model
lr_p = chi2.sf(lr_stat, lr_df)

# Extract p-values for age and site (joint)
# Joint test for site (all site coefficients)
site_terms = [name for name in model_main.params.index if name.startswith('C(site)')]
if site_terms:
    site_test = model_main.wald_test(' + '.join([f"{t}=0" for t in site_terms]))
    site_p = float(site_test.pvalue)
else:
    site_p = np.nan

age_p = float(model_main.pvalues['age'])

print('overall_rate', overall_rate)
print('site_rates', site_rates.to_dict())
print('age_rates', age_rates.to_dict())
print('age_p', age_p)
print('site_p', site_p)
print('interaction_lr_p', lr_p)
