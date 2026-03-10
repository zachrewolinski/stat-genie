import pandas as pd

df = pd.read_csv('reading.csv')

subset = df[['device','correct_rate']].dropna()
subset['device_gt0'] = subset['device'] > 0
print(subset['device_gt0'].value_counts())
print(pd.crosstab(subset['device_gt0'], subset['correct_rate']))

# compute agreement rate
agree = (subset['device_gt0'].astype(int) == subset['correct_rate'].astype(int)).mean()
print('agreement', agree)
