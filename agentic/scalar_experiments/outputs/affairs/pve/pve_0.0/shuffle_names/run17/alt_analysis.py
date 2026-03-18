import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('affairs.csv')

# children indicator (religiousness column yes/no)
child = df['religiousness'].map({'yes':1,'no':0})

for outcome in ['affairs','rating','age']:
    sub = pd.DataFrame({'child': child, 'outcome': df[outcome]}).dropna()
    vals_yes = sub[sub['child']==1]['outcome']
    vals_no = sub[sub['child']==0]['outcome']
    t_stat, p_val = stats.ttest_ind(vals_yes, vals_no, equal_var=False)
    mean_diff = vals_yes.mean() - vals_no.mean()
    print(outcome, 'mean_yes', vals_yes.mean(), 'mean_no', vals_no.mean(), 'diff', mean_diff, 'p', p_val)
