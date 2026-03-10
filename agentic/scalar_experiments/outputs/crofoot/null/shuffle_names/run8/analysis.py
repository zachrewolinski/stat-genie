import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Map columns based on info.json descriptions (names are shuffled)
# Outcome: m_focal (1 if focal wins)
# Relative group size: f_other (focal group size) vs win (other group size)
# Contest location: m_other (focal distance from its home range center)
#                   n_focal (other distance from its home range center)

# Ensure numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create predictors
size_diff = df['f_other'] - df['win']
size_ratio = df['f_other'] / df['win']
loc_diff = df['n_focal'] - df['m_other']  # positive => focal closer to its center than other
loc_ratio = df['m_other'] / df['n_focal']

# Outcome
y = df['m_focal']

# Logistic regression with size_diff and loc_diff
X = pd.DataFrame({
    'size_diff': size_diff,
    'loc_diff': loc_diff
})
X = sm.add_constant(X)
model = sm.Logit(y, X, missing='drop')
result = model.fit(disp=False)

# Alternative spec using ratios (log-transformed)
X2 = pd.DataFrame({
    'log_size_ratio': np.log(size_ratio),
    'log_loc_ratio': np.log(loc_ratio)
})
X2 = sm.add_constant(X2)
model2 = sm.Logit(y, X2, missing='drop')
result2 = model2.fit(disp=False)

# Simple correlations (point-biserial) for quick sense
pb_size = stats.pointbiserialr(y, size_diff)
pb_loc = stats.pointbiserialr(y, loc_diff)

print('N:', len(df))
print('\nLogit (diffs):')
print(result.summary())
print('\nLogit (log ratios):')
print(result2.summary())
print('\nPoint-biserial correlations:')
print('size_diff r=%.3f p=%.4f' % (pb_size.correlation, pb_size.pvalue))
print('loc_diff r=%.3f p=%.4f' % (pb_loc.correlation, pb_loc.pvalue))
