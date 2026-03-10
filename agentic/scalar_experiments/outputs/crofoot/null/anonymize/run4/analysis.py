import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Define variables
df = df.copy()
df['size_diff'] = df['feature7'] - df['feature8']
df['size_ratio'] = df['feature7'] / df['feature8']
df['loc_adv'] = df['feature6'] - df['feature5']  # positive means other farther from its center than focal

y = df['feature4']
X = df[['size_diff', 'loc_adv']]
X = sm.add_constant(X)

model = sm.Logit(y, X)
result = model.fit(disp=False)
print(result.summary())

# also show odds ratios and p-values
odds = np.exp(result.params)
print('\nOdds ratios:')
print(odds)
print('\nP-values:')
print(result.pvalues)

# alternative model with size_ratio and loc_adv
X2 = sm.add_constant(df[['size_ratio','loc_adv']])
model2 = sm.Logit(y, X2)
result2 = model2.fit(disp=False)
print('\nModel with size_ratio:')
print(result2.summary())
print('\nOdds ratios:')
print(np.exp(result2.params))
print('\nP-values:')
print(result2.pvalues)
