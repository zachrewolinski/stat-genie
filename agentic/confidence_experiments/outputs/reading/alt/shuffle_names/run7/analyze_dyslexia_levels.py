import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df=pd.read_csv('reading.csv')

for level in [1.0,2.0]:
    sub=df[(df['dyslexia']==level) & df['language'].notna() & df['running_time'].notna() & df['speed'].notna()]
    if len(sub)==0:
        print('level', level, 'no data')
        continue
    sub=sub.copy()
    sub['log_speed']=np.log(sub['running_time'])
    rv0=sub[sub['language']==0]['log_speed']
    rv1=sub[sub['language']==1]['log_speed']
    if len(rv0)>1 and len(rv1)>1:
        tt=stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
        mean_diff=rv1.mean()-rv0.mean()
        pooled_sd=np.sqrt(((rv1.var(ddof=1)+rv0.var(ddof=1))/2))
        d=mean_diff/pooled_sd
    else:
        tt=None
        d=np.nan
    print('\nDyslexia level', level, 'rows', len(sub), 'participants', sub['speed'].nunique())
    print('counts', sub['language'].value_counts())
    print('log-speed t-test', tt, 'd', d)
    if len(sub) > 10:
        model = smf.ols('log_speed ~ language', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['speed']})
        print(model.summary().tables[1])

