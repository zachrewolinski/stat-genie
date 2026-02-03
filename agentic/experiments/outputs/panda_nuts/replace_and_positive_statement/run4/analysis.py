import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF_PATH = 'panda_nuts.csv'

df = pd.read_csv(DF_PATH)

# Define nut-cracking efficiency as nuts opened per second
# (higher means more efficient)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Fit linear model with age (numeric) and sex/help (categorical)
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()

# Simple descriptive stats for context
summary_stats = df.groupby(['sex', 'help'])['efficiency'].agg(['count', 'mean', 'std']).reset_index()

print('Model summary:')
print(model.summary())
print('\nGroup-level efficiency stats (nuts/sec):')
print(summary_stats.to_string(index=False))
