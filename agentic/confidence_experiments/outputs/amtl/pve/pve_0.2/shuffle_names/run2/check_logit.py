import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

p = 1/(1+np.exp(-df['genus']))
exp_counts = p * df['age']
# measure closeness to integer
frac_dist = np.mean(np.abs(exp_counts - np.round(exp_counts)))
print('mean abs dist to nearest integer:', frac_dist)
print('min/max expected counts:', exp_counts.min(), exp_counts.max())
print('fraction within 0.2 of integer:', np.mean(np.abs(exp_counts - np.round(exp_counts)) < 0.2))
print('fraction within 0.05 of integer:', np.mean(np.abs(exp_counts - np.round(exp_counts)) < 0.05))

# check if logit transform of (num_amtl/age) matches genus
# maybe num_amtl is missing count? test
p2 = df['num_amtl'] / df['age']
logit_p2 = np.log(p2/(1-p2))
print('num_amtl/age within (0,1)?', np.mean((p2>0)&(p2<1)))
print('corr genus with logit(num_amtl/age):', np.corrcoef(df['genus'], logit_p2.replace([np.inf, -np.inf], np.nan).fillna(0))[0,1])
