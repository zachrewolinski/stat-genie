import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Clean/prepare
# children column is 'yes'/'no'
df['children_bin'] = df['children'].map({'yes': 1, 'no': 0})

# Basic group stats
summary = df.groupby('children')['affairs'].agg(['count', 'mean', 'median'])
summary_any = df.groupby('children').apply(lambda g: (g['affairs'] > 0).mean()).rename('prop_any_affair')

# t-test for difference in mean affairs
children_yes = df.loc[df['children'] == 'yes', 'affairs']
children_no = df.loc[df['children'] == 'no', 'affairs']

t_stat, p_val, _ = ttest_ind(children_yes, children_no, usevar='unequal')

# OLS with controls
# Use female indicator if present; gender is categorical, so prefer numeric female
control_cols = ['children_bin', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'female']
model_df = df[control_cols + ['affairs']].dropna()

X = sm.add_constant(model_df[control_cols])
y = model_df['affairs']
ols = sm.OLS(y, X).fit()

# Logistic regression on any affair
model_df['any_affair'] = (model_df['affairs'] > 0).astype(int)
logit = sm.Logit(model_df['any_affair'], X).fit(disp=False)

# Print results
print('Group summary (affairs):')
print(summary)
print('\nProportion with any affair:')
print(summary_any)
print('\nT-test (mean affairs, yes vs no):')
print({'t_stat': t_stat, 'p_value': p_val})

print('\nOLS coefficient for children (affairs ~ controls):')
print(ols.params['children_bin'])
print('OLS p-value:', ols.pvalues['children_bin'])

print('\nLogit coefficient for children (any affair ~ controls):')
print(logit.params['children_bin'])
print('Logit p-value:', logit.pvalues['children_bin'])
