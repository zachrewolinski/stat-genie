import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

p = 1/(1+np.exp(-df['feature3']))
miss_est = p*df['feature4']
print('miss_est min/max', miss_est.min(), miss_est.max())
print('miss_est round unique counts', len(np.unique(np.round(miss_est))))

m_est = np.round(p*(df['feature4']+1)-0.5)
pe = (m_est+0.5)/(df['feature4']+1)
logit_pe = np.log(pe/(1-pe))
print('m_est min/max', m_est.min(), m_est.max())
print('logit diff mean abs', np.mean(np.abs(logit_pe - df['feature3'])))

print('feature3 rounded range', df['feature3'].round().min(), df['feature3'].round().max())
print('corr feature3 with feature4', df['feature3'].corr(df['feature4']))

