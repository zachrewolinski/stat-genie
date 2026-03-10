import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Outcome
outcome = df['m_focal']

# Try swapped size columns
focal_size = df['win']
other_size = df['f_other']
rel_size = focal_size - other_size

# Distances swapped maybe? Try both ways
focal_dist = df['m_other']
other_dist = df['n_focal']
rel_dist = other_dist - focal_dist

analysis_df = df.copy()
analysis_df['rel_size'] = rel_size
analysis_df['rel_dist'] = rel_dist

model = smf.logit('m_focal ~ rel_size + rel_dist', data=analysis_df).fit(disp=False)
print('Swapped size columns:')
print(model.summary())

model_size = smf.logit('m_focal ~ rel_size', data=analysis_df).fit(disp=False)
model_dist = smf.logit('m_focal ~ rel_dist', data=analysis_df).fit(disp=False)
print('\nSize only:')
print(model_size.summary())
print('\nDist only:')
print(model_dist.summary())

# Now try swapped distances (if labels swapped)
focal_dist2 = df['n_focal']
other_dist2 = df['m_other']
rel_dist2 = other_dist2 - focal_dist2
analysis_df['rel_dist2'] = rel_dist2

model2 = smf.logit('m_focal ~ rel_size + rel_dist2', data=analysis_df).fit(disp=False)
print('\nSwapped distance definition:')
print(model2.summary())

model_dist2 = smf.logit('m_focal ~ rel_dist2', data=analysis_df).fit(disp=False)
print('\nDist2 only:')
print(model_dist2.summary())
