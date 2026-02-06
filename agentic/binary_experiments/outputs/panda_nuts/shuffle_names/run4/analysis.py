import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Map columns based on metadata mismatch
# age: numeric age in years
# nuts_opened: sex (m/f)
# seconds: help received (y/N)
# help: number of nuts opened in session
# chimpanzee: duration in seconds

_df = _df.copy()
_df['sex_cat'] = _df['nuts_opened'].astype('category')
_df['help_cat'] = _df['seconds'].astype('category')

# Efficiency: nuts opened per second
_df['efficiency'] = _df['help'] / _df['chimpanzee']

# Remove any invalid rows (e.g., zero duration)
_df = _df.replace([np.inf, -np.inf], np.nan).dropna(subset=['efficiency', 'age', 'sex_cat', 'help_cat'])

# Fit OLS with robust SE
model = smf.ols('efficiency ~ age + C(sex_cat) + C(help_cat)', data=_df).fit(cov_type='HC3')

# Save summary to a text file for inspection
with open('analysis_summary.txt', 'w') as f:
    f.write(model.summary().as_text())
    f.write('\n\n')
    f.write('N=' + str(len(_df)) + '\n')

# Also compute group means for interpretation
means = _df.groupby(['sex_cat', 'help_cat'])['efficiency'].mean().reset_index()
means.to_csv('group_means.csv', index=False)

print(model.summary())
print('\nGroup means (efficiency):')
print(means)
