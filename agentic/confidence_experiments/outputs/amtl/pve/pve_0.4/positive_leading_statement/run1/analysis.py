import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# Basic checks
num_amtl = df['num_amtl']
nearest_int = np.round(num_amtl)
frac_close_int = np.mean(np.isclose(num_amtl, nearest_int, atol=1e-6))
mean_abs_diff = np.mean(np.abs(num_amtl - nearest_int))

print('num_amtl min/max:', num_amtl.min(), num_amtl.max())
print('frac close to int:', frac_close_int)
print('mean abs diff to int:', mean_abs_diff)

# Check if num_amtl could be proportion or logit
prop_like = ((num_amtl >= 0) & (num_amtl <= 1)).mean()
print('fraction within [0,1]:', prop_like)

# If we inverse-logit to get p, see if p*sockets near int
p = 1 / (1 + np.exp(-num_amtl))
counts_est = p * df['sockets']
counts_nearest = np.round(counts_est)
frac_counts_close = np.mean(np.isclose(counts_est, counts_nearest, atol=1e-2))
print('frac inverse-logit counts close to int (<=0.01):', frac_counts_close)

# Fit OLS on num_amtl
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model.summary())

# Extract genus effects relative to reference
# statsmodels uses alphabetical reference for C(genus)

