import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Rename for clarity
# feature4: focal win (1) vs other win (0)
# feature5: distance of focal from center of its home range
# feature6: distance of other from center of its home range
# feature7: focal group size
# feature8: other group size

# Derived predictors
# Relative group size (focal - other). Positive means focal larger.
df['rel_size'] = df['feature7'] - df['feature8']

# Location advantage: other distance - focal distance.
# Positive means focal is closer to its home range center than the other group.
df['loc_adv'] = df['feature6'] - df['feature5']

# Outcome
y = df['feature4']
X = df[['rel_size', 'loc_adv']]
X = sm.add_constant(X)

# Fit logistic regression
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Also compute a simple descriptive win rate split by sign of predictors
win_rate_by_size = df.groupby(df['rel_size'] > 0)['feature4'].mean()
win_rate_by_loc = df.groupby(df['loc_adv'] > 0)['feature4'].mean()

# Print key outputs
print('Logit coefficients:')
print(result.params)
print('\nLogit p-values:')
print(result.pvalues)
print('\nOdds ratios:')
print(np.exp(result.params))

print('\nWin rate by relative size (focal larger?):')
print(win_rate_by_size)
print('\nWin rate by location advantage (focal closer to center?):')
print(win_rate_by_loc)

# Save a compact summary for reference
with open('analysis_summary.txt', 'w') as f:
    f.write(result.summary2().as_text())
    f.write('\n\nWin rate by relative size (focal larger?):\n')
    f.write(win_rate_by_size.to_string())
    f.write('\n\nWin rate by location advantage (focal closer to center?):\n')
    f.write(win_rate_by_loc.to_string())
