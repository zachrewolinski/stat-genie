import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')
missing_exp = np.exp(_df['genus'])
within = ((missing_exp >= 0) & (missing_exp <= _df['age'])).mean()
print('exp(genus) within 0..age fraction', within)
print('exp(genus) min/max', missing_exp.min(), missing_exp.max())

# check if exp(genus) near integer
frac_int = np.mean(np.isclose(missing_exp, np.round(missing_exp), atol=1e-2))
print('exp(genus) integer-like', frac_int)

# check if genus itself maybe missing count / age (proportion) * ???
prop = _df['genus'] / _df['age']
print('genus/age range', prop.min(), prop.max())

