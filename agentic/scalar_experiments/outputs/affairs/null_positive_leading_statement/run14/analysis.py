import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Basic cleaning
# Ensure children categorical
_df['children'] = _df['children'].astype('category')

# Outcome variables
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Group stats
summary = _df.groupby('children').agg(
    n=('affairs','size'),
    mean_affairs=('affairs','mean'),
    median_affairs=('affairs','median'),
    prop_any=('any_affair','mean')
)

# Simple difference in means
mean_yes = summary.loc['yes','mean_affairs']
mean_no = summary.loc['no','mean_affairs']
prop_yes = summary.loc['yes','prop_any']
prop_no = summary.loc['no','prop_any']

# t-test for means (Welch)
from scipy import stats

aff_yes = _df.loc[_df['children']=='yes','affairs']
aff_no = _df.loc[_df['children']=='no','affairs']

t_res = stats.ttest_ind(aff_yes, aff_no, equal_var=False)

# Logistic regression for any affair
logit = smf.logit('any_affair ~ C(children) + age + yearsmarried + C(gender) + religiousness + education + occupation + rating', data=_df).fit(disp=False)

# Poisson regression for count of affairs
poisson = smf.glm('affairs ~ C(children) + age + yearsmarried + C(gender) + religiousness + education + occupation + rating', data=_df, family=sm.families.Poisson()).fit()

# Extract effect of children yes (reference is no if category ordered)
# statsmodels uses first category alphabetically as base; children categories are ['no','yes']
coef_logit = logit.params.get('C(children)[T.yes]', np.nan)
se_logit = logit.bse.get('C(children)[T.yes]', np.nan)

coef_pois = poisson.params.get('C(children)[T.yes]', np.nan)
se_pois = poisson.bse.get('C(children)[T.yes]', np.nan)

# Convert to odds ratio and rate ratio
or_logit = np.exp(coef_logit)
rr_pois = np.exp(coef_pois)

p_logit = logit.pvalues.get('C(children)[T.yes]', np.nan)
p_pois = poisson.pvalues.get('C(children)[T.yes]', np.nan)

print('SUMMARY')
print(summary)
print('\nDIFF mean affairs (yes-no):', mean_yes - mean_no)
print('DIFF prop any (yes-no):', prop_yes - prop_no)
print('\nT-test:', t_res)
print('\nLogit coef children yes:', coef_logit, 'OR', or_logit, 'SE', se_logit, 'p', p_logit)
print('Poisson coef children yes:', coef_pois, 'RR', rr_pois, 'SE', se_pois, 'p', p_pois)
