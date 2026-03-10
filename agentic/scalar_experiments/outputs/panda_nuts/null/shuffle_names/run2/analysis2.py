import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('panda_nuts.csv')

df['efficiency'] = df['help'] / df['chimpanzee']

df['age_years'] = df['hammer']
df['sex'] = df['nuts_opened']
df['received_help'] = df['seconds']

# Spearman correlation for age vs efficiency
rho, p_rho = stats.spearmanr(df['age_years'], df['efficiency'])
print('Spearman age vs efficiency:', rho, p_rho)

# Mann-Whitney U for sex
g1 = df[df['sex']=='f']['efficiency']
g2 = df[df['sex']=='m']['efficiency']
if len(g1)>0 and len(g2)>0:
    u, p_u = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    print('Mann-Whitney sex (f vs m):', u, p_u)

# Mann-Whitney U for received help
h1 = df[df['received_help']=='y']['efficiency']
h0 = df[df['received_help']=='N']['efficiency']
if len(h1)>0 and len(h0)>0:
    u2, p_u2 = stats.mannwhitneyu(h1, h0, alternative='two-sided')
    print('Mann-Whitney help (y vs N):', u2, p_u2)

# Show group medians
print('Median efficiency by sex:')
print(df.groupby('sex')['efficiency'].median())
print('Median efficiency by received_help:')
print(df.groupby('received_help')['efficiency'].median())
