import pandas as pd
import numpy as np

path = 'amtl.csv'

df = pd.read_csv(path)
# suspect mapping
sockets = df['age']
logit = df['genus']
p = 1 / (1 + np.exp(-logit))
missing_est = p * sockets
print('missing_est summary', missing_est.describe())
print('missing_est min/max', missing_est.min(), missing_est.max())

# check if missing_est close to integer
close_int = np.mean(np.isclose(missing_est, np.round(missing_est), atol=0.05))
print('fraction missing_est close to int (0.05)', close_int)

# check if logit values correspond to logit of some proportion derived from num_amtl or pop etc
for col in ['num_amtl', 'pop']:
    vals = df[col]
    # if col is counts? compute logit proportion with sockets
    # avoid division by zero
    p_col = np.clip(vals / sockets, 1e-6, 1-1e-6)
    logit_col = np.log(p_col / (1 - p_col))
    corr = np.corrcoef(logit, logit_col)[0,1]
    print(f'corr logit vs logit({col}/sockets):', corr)

# check if genus maybe equals num_amtl - sockets or similar
print('corr genus vs num_amtl', np.corrcoef(df["genus"], df['num_amtl'])[0,1])
print('corr genus vs pop', np.corrcoef(df["genus"], df['pop'])[0,1])

