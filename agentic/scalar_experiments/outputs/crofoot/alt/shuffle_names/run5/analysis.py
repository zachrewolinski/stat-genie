import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Load data
path = Path('crofoot.csv')
df = pd.read_csv(path)

# Map columns to semantic meanings based on info.json descriptions
# m_focal: 1 if focal won contest, 0 if other won
# m_other: distance (m) of focal group from center of its home range
# n_focal: distance (m) of other group from center of its home range
# f_other: number of individuals in focal group
# win: number of individuals in other group

outcome = df['m_focal']

focal_size = df['f_other']
other_size = df['win']

# Relative group size (difference and ratio)
rel_size_diff = focal_size - other_size
rel_size_ratio = focal_size / other_size

# Contest location relative to home range centers
# Positive rel_loc_diff means other group is farther from its center than focal group
rel_loc_diff = df['n_focal'] - df['m_other']
rel_loc_ratio = df['m_other'] / df['n_focal']

# Build modeling dataframe
model_df = pd.DataFrame({
    'win': outcome,
    'rel_size_diff': rel_size_diff,
    'rel_size_ratio': rel_size_ratio,
    'rel_loc_diff': rel_loc_diff,
    'rel_loc_ratio': rel_loc_ratio,
    'dist_focal': df['m_other'],
    'dist_other': df['n_focal'],
    'focal_size': focal_size,
    'other_size': other_size,
})

# Drop any missing
model_df = model_df.dropna()

# Logistic regression: win ~ rel_size_diff + rel_loc_diff
X1 = sm.add_constant(model_df[['rel_size_diff', 'rel_loc_diff']])
logit1 = sm.Logit(model_df['win'], X1).fit(disp=False)

# Logistic regression: win ~ rel_size_ratio + rel_loc_diff
X2 = sm.add_constant(model_df[['rel_size_ratio', 'rel_loc_diff']])
logit2 = sm.Logit(model_df['win'], X2).fit(disp=False)

# Logistic regression: win ~ rel_size_diff + dist_focal + dist_other
X3 = sm.add_constant(model_df[['rel_size_diff', 'dist_focal', 'dist_other']])
logit3 = sm.Logit(model_df['win'], X3).fit(disp=False)

# Summaries
print('N:', len(model_df))
print('\nModel 1: win ~ rel_size_diff + rel_loc_diff')
print(logit1.summary())
print('\nModel 2: win ~ rel_size_ratio + rel_loc_diff')
print(logit2.summary())
print('\nModel 3: win ~ rel_size_diff + dist_focal + dist_other')
print(logit3.summary())

# Also compute simple correlations and group-level means for context
print('\nDescriptive stats:')
print(model_df[['rel_size_diff', 'rel_loc_diff', 'dist_focal', 'dist_other']].describe())

# Check win rates by bins of rel_size_diff and rel_loc_diff
model_df['rel_size_bin'] = pd.qcut(model_df['rel_size_diff'], q=3, duplicates='drop')
model_df['rel_loc_bin'] = pd.qcut(model_df['rel_loc_diff'], q=3, duplicates='drop')

win_by_size = model_df.groupby('rel_size_bin')['win'].mean()
win_by_loc = model_df.groupby('rel_loc_bin')['win'].mean()

print('\nWin rate by rel_size_diff tercile:')
print(win_by_size)
print('\nWin rate by rel_loc_diff tercile:')
print(win_by_loc)
