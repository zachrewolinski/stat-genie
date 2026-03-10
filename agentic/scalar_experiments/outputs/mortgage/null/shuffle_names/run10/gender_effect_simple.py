import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

_df=pd.read_csv('mortgage.csv')
# Based on metadata descriptions:
# - 'denied_PMI' column represents female (1=female, 0=male)
# - 'deny' column represents acceptance (1=accepted, 0=denied)

female=_df['denied_PMI']
approve=_df['deny']

# drop missing gender or approval
mask=~female.isna() & ~approve.isna()
female=female[mask]
approve=approve[mask]

ct=pd.crosstab(female, approve)
print('crosstab (female x approve):')
print(ct)

# acceptance rate by gender
rates=ct.div(ct.sum(axis=1), axis=0)
print('acceptance rate by gender:')
print(rates[1])

# chi-square test
chi2,p,_,_ = chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# difference in acceptance rates
rate_f=rates.loc[1,1] if 1 in rates.index else float('nan')
rate_m=rates.loc[0,1] if 0 in rates.index else float('nan')
print('rate_female', rate_f, 'rate_male', rate_m, 'diff', rate_f-rate_m)

