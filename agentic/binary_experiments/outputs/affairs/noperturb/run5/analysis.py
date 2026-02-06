import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Load data
DF_PATH = 'affairs.csv'
df = pd.read_csv(DF_PATH)

# Basic derived variables

df['any_affair'] = (df['affairs'] > 0).astype(int)

# Group summaries
summary = df.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    prop_any_affair=('any_affair', 'mean')
)

# Difference in means (affairs count)
no_affairs = df.loc[df['children'] == 'no', 'affairs']
yes_affairs = df.loc[df['children'] == 'yes', 'affairs']
t_stat, p_val, _ = ttest_ind(yes_affairs, no_affairs, usevar='unequal')

# Logistic regression for any affair
logit_model = smf.logit(
    'any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)',
    data=df
).fit(disp=0)

coef = logit_model.params.get('C(children)[T.yes]', np.nan)
se = logit_model.bse.get('C(children)[T.yes]', np.nan)
p_logit = logit_model.pvalues.get('C(children)[T.yes]', np.nan)
odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

# Average marginal effect for children
margeff = logit_model.get_margeff(at='overall')
me_table = margeff.summary_frame()
me_children = me_table.loc['C(children)[T.yes]']

print('Group summary by children')
print(summary)
print('\nT-test (affairs count) yes vs no children:')
print({'t_stat': float(t_stat), 'p_value': float(p_val)})
print('\nLogit coefficient for children=yes (baseline no):')
print({'coef': float(coef), 'se': float(se), 'p_value': float(p_logit), 'odds_ratio': odds_ratio})
print('\nAverage marginal effect (children=yes):')
print(me_children)
