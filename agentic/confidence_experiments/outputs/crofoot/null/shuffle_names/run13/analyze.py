import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Map variables based on info.json descriptions
# Outcome: focal won (1) vs other (0)
win = df['m_focal']

# Group sizes
focal_size = df['f_other']  # number of individuals in focal group
other_size = df['win']      # number of individuals in other group

# Distances from respective home range centers
focal_dist = df['m_other']  # distance of focal group from its home range center
other_dist = df['n_focal']  # distance of other group from its home range center

# Derived predictors
rel_size = focal_size - other_size
# Positive means contest closer to focal group relative to other
rel_location = other_dist - focal_dist

analysis_df = pd.DataFrame({
    'win': win,
    'rel_size': rel_size,
    'rel_location': rel_location,
    'focal_size': focal_size,
    'other_size': other_size,
    'focal_dist': focal_dist,
    'other_dist': other_dist,
})

# Drop any missing values (should be none)
analysis_df = analysis_df.dropna()

# Standardize predictors for comparable coefficients
for col in ['rel_size', 'rel_location']:
    analysis_df[col + '_z'] = (analysis_df[col] - analysis_df[col].mean()) / analysis_df[col].std(ddof=0)

X = analysis_df[['rel_size_z', 'rel_location_z']]
X = sm.add_constant(X)

model = sm.Logit(analysis_df['win'], X).fit(disp=False)

# Also fit model with focal and other sizes separately to check robustness
X2 = analysis_df[['focal_size', 'other_size', 'focal_dist', 'other_dist']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(analysis_df['win'], X2).fit(disp=False)

# Summaries
print('N:', len(analysis_df))
print('\nLogit: win ~ rel_size_z + rel_location_z')
print(model.summary())

print('\nOdds ratios (1 SD change):')
params = model.params
conf = model.conf_int()
for var in ['rel_size_z', 'rel_location_z']:
    or_val = np.exp(params[var])
    or_lo = np.exp(conf.loc[var, 0])
    or_hi = np.exp(conf.loc[var, 1])
    print(f"{var}: OR={or_val:.3f}, 95% CI [{or_lo:.3f}, {or_hi:.3f}], p={model.pvalues[var]:.4f}")

print('\nLogit: win ~ focal_size + other_size + focal_dist + other_dist')
print(model2.summary())

# Simple correlations for context
print('\nCorrelations (point-biserial):')
for col in ['rel_size', 'rel_location', 'focal_size', 'other_size', 'focal_dist', 'other_dist']:
    r = np.corrcoef(analysis_df['win'], analysis_df[col])[0,1]
    print(f"win vs {col}: r={r:.3f}")
