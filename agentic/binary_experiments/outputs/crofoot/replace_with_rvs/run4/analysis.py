import pandas as pd
import statsmodels.api as sm

# Load data

df = pd.read_csv('crofoot.csv')

# Create predictors
# Relative group size (focal minus other)

df['rel_size'] = df['n_focal'] - df['n_other']

# Relative contest location: how much farther the other group is from its home-range center
# Positive means focal is closer to its own center than the other group is to its own center

df['rel_dist'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for interpretability
for col in ['rel_size', 'rel_dist']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression: win ~ relative size + relative location
X = df[['rel_size_z', 'rel_dist_z']]
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X)
result = model.fit(disp=False)

print(result.summary())

# Also show coefficients and p-values
print('\nCoefficients:')
print(result.params)

print('\nP-values:')
print(result.pvalues)

# Simple descriptive: win rates by sign of rel_size and rel_dist

df['rel_size_pos'] = df['rel_size'] > 0
df['rel_dist_pos'] = df['rel_dist'] > 0

print('\nWin rate by rel_size_pos:')
print(df.groupby('rel_size_pos')['win'].mean())

print('\nWin rate by rel_dist_pos:')
print(df.groupby('rel_dist_pos')['win'].mean())
