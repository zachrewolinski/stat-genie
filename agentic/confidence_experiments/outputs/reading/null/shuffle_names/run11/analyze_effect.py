import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df=pd.read_csv('reading.csv')

# assume reader_view indicator is 'language' (0/1)
reader_var='language'
# assume reading speed is 'running_time'
speed_var='running_time'

# candidate dyslexia columns
candidates=['device','dyslexia']

results=[]

for dys in candidates:
    sub=df[[reader_var, speed_var, dys]].dropna()
    # dyslexic group define as >0
    dys_sub=sub[sub[dys]>0]
    if dys_sub.empty:
        continue
    # split
    g0=dys_sub[dys_sub[reader_var]==0][speed_var]
    g1=dys_sub[dys_sub[reader_var]==1][speed_var]
    # t-test with unequal variance
    tstat, pval=stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
    # effect size (Cohen's d)
    def cohens_d(a,b):
        a=a.dropna(); b=b.dropna()
        n1=len(a); n2=len(b)
        s1=a.var(ddof=1); s2=b.var(ddof=1)
        if n1<2 or n2<2:
            return np.nan
        s_p=np.sqrt(((n1-1)*s1+(n2-1)*s2)/(n1+n2-2))
        return (a.mean()-b.mean())/s_p
    d=cohens_d(g1,g0)
    # mann-whitney
    try:
        ustat, up=stats.mannwhitneyu(g1, g0, alternative='two-sided')
    except ValueError:
        ustat, up=np.nan, np.nan
    # regression with log(speed) to reduce skew
    dys_sub = dys_sub.copy()
    dys_sub['log_speed']=np.log(dys_sub[speed_var])
    model = smf.ols('log_speed ~ C('+reader_var+')', data=dys_sub).fit()
    coef = model.params.get('C('+reader_var+')[T.1]', np.nan)
    p_reg = model.pvalues.get('C('+reader_var+')[T.1]', np.nan)
    results.append((dys, len(dys_sub), len(g0), len(g1), g0.mean(), g1.mean(), tstat, pval, d, up, coef, p_reg))

print('reader_var', reader_var, 'speed_var', speed_var)
for r in results:
    dys, n, n0, n1, mean0, mean1, tstat, pval, d, up, coef, p_reg=r
    print('\nDyslexia column:', dys)
    print('n total', n, 'n0', n0, 'n1', n1)
    print('mean speed reader_view=0', mean0, 'reader_view=1', mean1)
    print('tstat', tstat, 'pval', pval)
    print('cohen d', d)
    print('mannwhitney p', up)
    print('log-speed coef', coef, 'p', p_reg)

# Also check non-dyslexic for reference
for dys in candidates:
    sub=df[[reader_var, speed_var, dys]].dropna()
    nd_sub=sub[sub[dys]==0]
    if nd_sub.empty:
        continue
    g0=nd_sub[nd_sub[reader_var]==0][speed_var]
    g1=nd_sub[nd_sub[reader_var]==1][speed_var]
    tstat, pval=stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
    print('\nNon-dyslexic comparison using', dys)
    print('mean0', g0.mean(), 'mean1', g1.mean(), 'p', pval)

