import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

path = Path('/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/crofoot/alt/anonymize/run6/crofoot.csv')

df = pd.read_csv(path)

# Outcome
y = df['feature4']

# Relative group size (focal - other)
relative_size = df['feature7'] - df['feature8']

# Location advantage: positive means contest is closer to focal group's home range center
location_adv = df['feature6'] - df['feature5']

# Build design matrix with intercept
X = pd.DataFrame({
    'relative_size': relative_size,
    'location_adv': location_adv,
})
X = sm.add_constant(X)

# Fit logistic regression
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Also fit model with standardized predictors for effect size comparability
X_std = X.copy()
for col in ['relative_size', 'location_adv']:
    X_std[col] = (X_std[col] - X_std[col].mean()) / X_std[col].std(ddof=0)
model_std = sm.Logit(y, X_std)
result_std = model_std.fit(disp=False)

# Compute predicted probability differences for 1 SD change
margeff = result_std.get_margeff(at='mean')

print('Logit results (raw predictors):')
print(result.summary())
print('\nLogit results (standardized predictors):')
print(result_std.summary())
print('\nMarginal effects at mean (standardized):')
print(margeff.summary())

# Simple descriptive checks
print('\nDescriptive:')
print('Mean win rate:', y.mean())
print('Relative size mean:', relative_size.mean(), 'std:', relative_size.std(ddof=0))
print('Location advantage mean:', location_adv.mean(), 'std:', location_adv.std(ddof=0))

# Optional: non-parametric check with focal closer indicator
focal_closer = (df['feature5'] < df['feature6']).astype(int)
X2 = sm.add_constant(pd.DataFrame({
    'relative_size': relative_size,
    'focal_closer': focal_closer,
}))
res2 = sm.Logit(y, X2).fit(disp=False)
print('\nLogit with focal_closer indicator:')
print(res2.summary())
