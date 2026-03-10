import pandas as pd
import numpy as np

path = '/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/reading/alt/anonymize/run4/reading.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print('rows', len(df))

# Identify candidate reading speed columns: numeric with large values maybe words/min
# Basic stats for feature20, and computed wpm from feature7 / feature5 etc.
for col in ['feature4','feature5','feature6','feature7','feature20']:
    if col in df.columns:
        print(col, df[col].describe())

# Compute reading speed as words per minute using feature5 (reading time minus scrolling)
# Avoid division by zero
read_time_min = df['feature5'] / 60000.0
wpm = df['feature7'] / read_time_min
print('computed_wpm_desc', wpm.replace([np.inf, -np.inf], np.nan).describe())

# Compare feature20 to computed wpm via correlation
if 'feature20' in df.columns:
    corr = df[['feature20']].join(wpm.rename('wpm')).corr().iloc[0,1]
    print('corr_feature20_wpm', corr)

# Check relation question: dyslexia status feature17 (1 yes) and reader view feature3 (1) on reading speed
# We'll compute mean wpm and mean feature20 by group
for speed_col in ['feature20']:
    if speed_col in df.columns:
        print('group means', speed_col)
        print(df.groupby(['feature17','feature3'])[speed_col].mean())

print('group means wpm')
print(df.assign(wpm=wpm).groupby(['feature17','feature3'])['wpm'].mean())

# Count observations
print('counts', df.groupby(['feature17','feature3']).size())
