import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Rename columns based on metadata descriptions and observed values
# 'nuts_opened' column holds sex (f/m)
# 'help' column holds number of nuts opened
# 'chimpanzee' column holds duration in seconds
# 'seconds' column holds help indicator (y/N)

df = df.rename(columns={
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened_count',
    'chimpanzee': 'duration_seconds',
    'seconds': 'helped'
})

# Map help indicator
help_map = {'y': 1, 'Y': 1, 'yes': 1, 'Yes': 1, 'N': 0, 'n': 0, 'no': 0, 'No': 0}
df['helped'] = df['helped'].map(help_map)

# Efficiency: nuts opened per second
# Avoid divide by zero (not expected)
df['efficiency'] = df['nuts_opened_count'] / df['duration_seconds']

print('Rows:', len(df))
print('Helped value counts:', df['helped'].value_counts(dropna=False).to_dict())
print('Sex value counts:', df['sex'].value_counts(dropna=False).to_dict())
print('Efficiency summary:\n', df['efficiency'].describe())
print('Missing values:', df[['age','sex','helped','nuts_opened_count','duration_seconds']].isna().sum().to_dict())

# Poisson regression with offset for duration
formula = 'nuts_opened_count ~ age + C(sex) + helped'
poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=np.log(df['duration_seconds']))
poisson_res = poisson_model.fit()
print('\nPoisson GLM results:')
print(poisson_res.summary())

# Overdispersion check
pearson_chi2 = poisson_res.pearson_chi2
df_resid = poisson_res.df_resid
overdisp = pearson_chi2 / df_resid
print(f'Overdispersion ratio (Pearson chi2 / df_resid): {overdisp:.3f}')

# Negative Binomial (if overdispersion)
nb_res = None
try:
    nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=np.log(df["duration_seconds"]))
    nb_res = nb_model.fit()
    print('\nNegative Binomial GLM results:')
    print(nb_res.summary())
except Exception as e:
    print('Negative Binomial fit failed:', e)

# Efficiency group summaries
print('\nEfficiency by sex:')
print(df.groupby('sex')['efficiency'].agg(['mean','median','count']))

print('\nEfficiency by helped:')
print(df.groupby('helped')['efficiency'].agg(['mean','median','count']))

# Correlation age vs efficiency (Spearman)
rho, pval = spearmanr(df['age'], df['efficiency'])
print(f'\nSpearman correlation age vs efficiency: rho={rho:.3f}, p={pval:.4f}')
