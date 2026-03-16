import pandas as pd
from scipy.stats import chi2_contingency

_df=pd.read_csv('mortgage.csv')

female=_df['denied_PMI']  # per metadata description

deny=_df['self_employed']  # per metadata description (1=denied)

mask=~female.isna() & ~deny.isna()
female=female[mask]
deny=deny[mask]

ct=pd.crosstab(female, deny)
print('crosstab (female x deny):')
print(ct)

rates=ct.div(ct.sum(axis=1), axis=0)
print('denial rate by gender:')
print(rates[1])

chi2,p,_,_ = chi2_contingency(ct)
print('chi2', chi2, 'p', p)

rate_f=rates.loc[1,1] if 1 in rates.index else float('nan')
rate_m=rates.loc[0,1] if 0 in rates.index else float('nan')
print('rate_female', rate_f, 'rate_male', rate_m, 'diff', rate_f-rate_m)

