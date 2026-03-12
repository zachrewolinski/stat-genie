import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# Load data
path = Path('panda_nuts.csv')
df = pd.read_csv(path)

# Map shuffled columns to semantic variables based on info.json metadata
# age column appears to be chimpanzee ID; hammer column appears to be age (years)
# nuts_opened column appears to be sex; sex column appears to be hammer type (not used here)
# help column appears to be number of nuts opened; chimpanzee column appears to be session duration in seconds
# seconds column appears to be whether help was received (y/N)

rename_map = {
    'age': 'chimp_id',
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened',
    'chimpanzee': 'session_seconds',
    'seconds': 'helped'
}

df = df.rename(columns=rename_map)

# Clean / recode
help_map = {'y': 1, 'Y': 1, 'yes': 1, 'Yes': 1, 'N': 0, 'n': 0, 'no': 0, 'No': 0}
df['helped_bin'] = df['helped'].map(help_map)

# efficiency: nuts opened per second
# avoid division by zero (if any) by keeping as NaN

df['efficiency'] = df['nuts_opened'] / df['session_seconds']

# basic summaries
print('Rows:', len(df))
print('Efficiency summary:', df['efficiency'].describe())
print('Helped counts:', df['helped'].value_counts())
print('Sex counts:', df['sex'].value_counts())

# OLS with robust SE (HC3)
model = smf.ols('efficiency ~ age_years + C(sex) + helped_bin', data=df).fit(cov_type='HC3')
print(model.summary())

# Also try log efficiency (add small constant)
min_pos = df.loc[df['efficiency'] > 0, 'efficiency'].min()
if pd.notna(min_pos):
    const = min_pos / 2
    df['log_eff'] = np.log(df['efficiency'] + const)
    model_log = smf.ols('log_eff ~ age_years + C(sex) + helped_bin', data=df).fit(cov_type='HC3')
    print('\nLOG MODEL:')
    print(model_log.summary())
else:
    print('No positive efficiency values found for log model')

# Group means for helped and sex
print('\nGroup means (efficiency):')
print(df.groupby('helped')['efficiency'].mean())
print(df.groupby('sex')['efficiency'].mean())

# Correlation with age
print('\nCorrelation efficiency vs age:', df['efficiency'].corr(df['age_years']))
