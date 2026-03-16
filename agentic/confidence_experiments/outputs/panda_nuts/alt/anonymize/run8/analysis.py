import pandas as pd
import statsmodels.formula.api as smf

# Load dataset
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Rename for clarity
col_map = {
    'feature2': 'age',
    'feature3': 'sex',
    'feature5': 'nuts_opened',
    'feature6': 'duration_sec',
    'feature7': 'help'
}

df = df.rename(columns=col_map)

# Efficiency: nuts per minute
# Use per-minute for interpretability; proportional to per-second so inference unchanged.
df['efficiency'] = df['nuts_opened'] / df['duration_sec'] * 60

# Clean categories
# Standardize help to y/N maybe

df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

# Fit linear regression
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

print('N rows:', len(df))
print('Efficiency summary:')
print(df['efficiency'].describe())
print('\nGroup means:')
print(df.groupby('sex')['efficiency'].mean())
print(df.groupby('help')['efficiency'].mean())
print('\nModel summary:')
print(model.summary())

# Save key stats
params = model.params
pvalues = model.pvalues

print('\nCoefficients:')
for k in params.index:
    print(k, params[k], 'p', pvalues[k])
