import pandas as pd
from scipy import stats

_df = pd.read_csv('affairs.csv')

affairs = _df['age']
children = _df['religiousness']
children_bool = children.str.lower().map({'yes': True, 'no': False})

summary = _df.groupby(children_bool)['age'].agg(['count', 'mean', 'median'])

any_affair = (affairs > 0).astype(int)
any_affair.name = 'any_affair'
any_summary = any_affair.groupby(children_bool).agg(['mean', 'count'])

print('Affair frequency by children (True=has children):')
print(summary)
print('\nAny affair (>0) rate by children:')
print(any_summary)

mean_diff = summary.loc[True, 'mean'] - summary.loc[False, 'mean']
rate_diff = any_summary.loc[True, 'mean'] - any_summary.loc[False, 'mean']

print('\nMean difference (children - no children):', mean_diff)
print('Any affair rate difference:', rate_diff)

with_children = affairs[children_bool]
without_children = affairs[~children_bool]
res = stats.ttest_ind(with_children, without_children, equal_var=False)
print('\nWelch t-test:', res)
