import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')

# check if genus between 0 and age
within = (( _df['genus'] >= 0) & (_df['genus'] <= _df['age'])).mean()
print('genus between 0 and age fraction', within)
print('min genus', _df['genus'].min(), 'max', _df['genus'].max())

# check if genus between 0 and age for each row count
neg = (_df['genus'] < 0).sum()
print('neg genus', neg)

# check if genus maybe logit of proportion missing (genus = logit(p))? then p=1/(1+exp(-genus))
# check if p*age close to integer? maybe missing count
p = 1/(1+np.exp(-_df['genus']))
missing_est = p * _df['age']
frac_int = np.mean(np.isclose(missing_est, np.round(missing_est), atol=1e-2))
print('missing_est integer-like fraction', frac_int)
print('missing_est range', missing_est.min(), missing_est.max())

# check if genus maybe log(missing+1)??
missing_exp = np.exp(_df['genus'])
print('exp(genus) range', missing_exp.min(), missing_exp.max())

# compare genus to num_amtl, maybe normalized by sockets? maybe genus is standardized missing counts

