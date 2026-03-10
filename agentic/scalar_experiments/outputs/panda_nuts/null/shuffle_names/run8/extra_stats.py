import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('panda_nuts.csv')

df = df.rename(columns={
    'help': 'nuts_opened_count',
    'chimpanzee': 'seconds',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'seconds': 'help_received',
})

df = df[df['seconds'] > 0].copy()

df['efficiency'] = df['nuts_opened_count'] / df['seconds']

# Spearman correlation age vs efficiency
rho, p_rho = stats.spearmanr(df['age'], df['efficiency'])

# Mann-Whitney for sex
sex_f = df[df['sex'] == 'f']['efficiency']
sex_m = df[df['sex'] == 'm']['efficiency']

u_sex, p_sex = stats.mannwhitneyu(sex_f, sex_m, alternative='two-sided')

# Mann-Whitney for help
help_y = df[df['help_received'].str.lower() == 'y']['efficiency']
help_n = df[df['help_received'].str.lower() == 'n']['efficiency']

u_help, p_help = stats.mannwhitneyu(help_y, help_n, alternative='two-sided')

print('Spearman age-efficiency: rho', rho, 'p', p_rho)
print('Mann-Whitney sex f vs m: U', u_sex, 'p', p_sex, 'n_f', len(sex_f), 'n_m', len(sex_m))
print('Mann-Whitney help y vs n: U', u_help, 'p', p_help, 'n_y', len(help_y), 'n_n', len(help_n))
