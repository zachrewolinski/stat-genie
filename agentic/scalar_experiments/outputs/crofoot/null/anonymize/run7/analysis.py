import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')

# Rename for clarity
outcome = df['feature4']  # 1 if focal won
focal_dist = df['feature5']
other_dist = df['feature6']
focal_size = df['feature7']
other_size = df['feature8']

# Construct predictors
rel_size = focal_size - other_size  # positive => focal larger
# location advantage: positive if contest closer to focal home range center
loc_adv = other_dist - focal_dist

# Also relative location index (0-1): focal_dist / (focal_dist + other_dist)
rel_loc_index = focal_dist / (focal_dist + other_dist)

# Assemble dataframe
analysis_df = pd.DataFrame({
    'outcome': outcome,
    'rel_size': rel_size,
    'loc_adv': loc_adv,
    'rel_loc_index': rel_loc_index,
    'focal_dist': focal_dist,
    'other_dist': other_dist
})

# Logistic regression: outcome ~ rel_size + loc_adv
X1 = sm.add_constant(analysis_df[['rel_size', 'loc_adv']])
model1 = sm.Logit(analysis_df['outcome'], X1).fit(disp=False)

# Logistic regression: outcome ~ rel_size + rel_loc_index
X2 = sm.add_constant(analysis_df[['rel_size', 'rel_loc_index']])
model2 = sm.Logit(analysis_df['outcome'], X2).fit(disp=False)

# Logistic regression: outcome ~ focal_dist + other_dist + rel_size
X3 = sm.add_constant(analysis_df[['rel_size', 'focal_dist', 'other_dist']])
model3 = sm.Logit(analysis_df['outcome'], X3).fit(disp=False)

# Simple comparisons for effect sizes
summary = {
    'n': len(analysis_df),
    'win_rate': analysis_df['outcome'].mean(),
    'rel_size_mean': analysis_df['rel_size'].mean(),
    'loc_adv_mean': analysis_df['loc_adv'].mean(),
}

# Compute group means by win/loss
means_by_outcome = analysis_df.groupby('outcome')[['rel_size','loc_adv','rel_loc_index','focal_dist','other_dist']].mean()

# Save results to a simple text output for inspection
print('SUMMARY')
print(summary)
print('\nMEANS BY OUTCOME')
print(means_by_outcome)

print('\nMODEL1: outcome ~ rel_size + loc_adv')
print(model1.summary())

print('\nMODEL2: outcome ~ rel_size + rel_loc_index')
print(model2.summary())

print('\nMODEL3: outcome ~ rel_size + focal_dist + other_dist')
print(model3.summary())

# Also compute odds ratios for model1
params = model1.params
conf = model1.conf_int()
conf.columns = ['2.5%','97.5%']

odds = pd.DataFrame({
    'coef': params,
    'odds_ratio': np.exp(params),
    'p_value': model1.pvalues,
    'ci_low': np.exp(conf['2.5%']),
    'ci_high': np.exp(conf['97.5%']),
})
print('\nODDS RATIOS MODEL1')
print(odds)

