import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Load data
DF = pd.read_csv('affairs.csv')

# Basic derived variables
DF['any_affair'] = (DF['affairs'] > 0).astype(int)

# Group summaries
group_stats = DF.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    mean_any_affair=('any_affair', 'mean')
)

# Two-sample t-test for mean affairs (unequal var)
with_children = DF.loc[DF['children'] == 'yes', 'affairs']
without_children = DF.loc[DF['children'] == 'no', 'affairs']

t_stat, p_val, dfree = ttest_ind(with_children, without_children, usevar='unequal')

# Regression: affairs count as continuous (OLS) with controls
ols = smf.ols(
    'affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=DF
).fit(cov_type='HC3')

# Logistic regression: any affair
logit = smf.logit(
    'any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=DF
).fit(disp=False)

# Extract children effect
ols_coef = ols.params.get('C(children)[T.yes]', np.nan)
ols_p = ols.pvalues.get('C(children)[T.yes]', np.nan)
logit_coef = logit.params.get('C(children)[T.yes]', np.nan)
logit_p = logit.pvalues.get('C(children)[T.yes]', np.nan)

# Print results
print('Group stats by children:')
print(group_stats)
print('\nT-test (affairs mean, yes vs no):')
print({'t_stat': float(t_stat), 'p_val': float(p_val), 'df': float(dfree)})
print('\nOLS (affairs) children effect:')
print({'coef': float(ols_coef), 'p_val': float(ols_p)})
print('\nLogit (any_affair) children effect:')
print({'coef': float(logit_coef), 'p_val': float(logit_p)})
