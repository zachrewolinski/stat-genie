import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Rename columns based on metadata mismatch
# age column seems to be individual ID; hammer column holds age in years
# nuts_opened column holds sex (m/f); sex column holds hammer type
# help column holds number of nuts opened; chimpanzee column holds session duration (seconds)
# seconds column holds whether help was received (y/N)
df = df.rename(columns={
    'age': 'chimp_id',
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened',
    'chimpanzee': 'session_seconds',
    'seconds': 'help_received'
})

# Clean help_received to binary
# values appear as 'y'/'N'
df['help_received'] = df['help_received'].str.strip().str.lower().map({'y': 1, 'n': 0})

# Some rows may be missing or unexpected values
# Drop rows with missing key fields
key_cols = ['age_years', 'sex', 'help_received', 'nuts_opened', 'session_seconds']
df = df.dropna(subset=key_cols)

# Ensure numeric types
for col in ['age_years', 'nuts_opened', 'session_seconds']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with non-positive session time
# (offset requires positive exposure)
df = df[df['session_seconds'] > 0]

# Poisson regression with offset for exposure time to model efficiency (rate of nuts opened)
# Predictors: age, sex, help_received
model = smf.glm(
    formula='nuts_opened ~ age_years + C(sex) + help_received',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['session_seconds'])
).fit()

print(model.summary())

# Save key results for interpretation
params = model.params
pvalues = model.pvalues
results = pd.DataFrame({'coef': params, 'pvalue': pvalues})
results.to_csv('model_results.csv')

# Compute rate ratios for interpretability
rate_ratios = np.exp(params)
rr = pd.DataFrame({'rate_ratio': rate_ratios, 'pvalue': pvalues})
rr.to_csv('rate_ratios.csv')

# Also compute simple group summaries for context
summary = df.groupby(['sex', 'help_received']).apply(
    lambda x: pd.Series({
        'mean_rate': (x['nuts_opened'] / x['session_seconds']).mean(),
        'median_rate': (x['nuts_opened'] / x['session_seconds']).median(),
        'n': len(x)
    })
).reset_index()
summary.to_csv('group_rate_summary.csv', index=False)
