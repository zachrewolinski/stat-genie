import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('crofoot.csv')

# Derived variables
rel_size = df['n_focal'] - df['n_other']
rel_dist = df['dist_focal'] - df['dist_other']  # positive if focal farther from its center

# Indicators
larger = rel_size > 0
closer = df['dist_focal'] < df['dist_other']  # contest closer to focal than other

# Win rates
summary = {}
summary['win_rate_overall'] = df['win'].mean()
summary['win_rate_larger'] = df.loc[larger, 'win'].mean()
summary['win_rate_not_larger'] = df.loc[~larger, 'win'].mean()
summary['n_larger'] = int(larger.sum())
summary['n_not_larger'] = int((~larger).sum())

summary['win_rate_closer'] = df.loc[closer, 'win'].mean()
summary['win_rate_not_closer'] = df.loc[~closer, 'win'].mean()
summary['n_closer'] = int(closer.sum())
summary['n_not_closer'] = int((~closer).sum())

print(summary)

# Fisher exact test for 2x2 (larger vs win)
# Build contingency tables
# rows: larger (1/0), cols: win (1/0)

cont_larger = pd.crosstab(larger, df['win'])
print('cont_larger')
print(cont_larger)

# ensure 2x2
if cont_larger.shape == (2,2):
    oddsratio, p = stats.fisher_exact(cont_larger)
    print('fisher larger p', p, 'oddsratio', oddsratio)

cont_closer = pd.crosstab(closer, df['win'])
print('cont_closer')
print(cont_closer)
if cont_closer.shape == (2,2):
    oddsratio2, p2 = stats.fisher_exact(cont_closer)
    print('fisher closer p', p2, 'oddsratio', oddsratio2)

