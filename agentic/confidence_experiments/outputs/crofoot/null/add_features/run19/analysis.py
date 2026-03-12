import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Basic cleaning: drop rows with missing in relevant columns
cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
df = df.dropna(subset=cols).copy()

# Create predictors
# Relative group size: difference in group size
# Location advantage: positive means contest closer to focal group's home range center
# (other farther from its own center than focal is from its center)
df['rel_size'] = df['n_focal'] - df['n_other']
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for interpretability and to compare effect sizes
for c in ['rel_size', 'loc_adv']:
    df[c + '_z'] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

# Logistic regression
model = smf.glm('win ~ rel_size_z + loc_adv_z', data=df, family=sm.families.Binomial()).fit()

# Add interaction as sensitivity check
model_int = smf.glm('win ~ rel_size_z * loc_adv_z', data=df, family=sm.families.Binomial()).fit()

# Extract results
summary = model.summary2().tables[1]
summary_int = model_int.summary2().tables[1]

# Predicted probabilities for low/high values (1 SD) for interpretability
# Baseline at mean predictors
baseline = model.predict(pd.DataFrame({'rel_size_z':[0], 'loc_adv_z':[0]})).iloc[0]
# Effects of +1 SD
pred_rel = model.predict(pd.DataFrame({'rel_size_z':[1], 'loc_adv_z':[0]})).iloc[0]
pred_loc = model.predict(pd.DataFrame({'rel_size_z':[0], 'loc_adv_z':[1]})).iloc[0]
# Effects of -1 SD
pred_rel_neg = model.predict(pd.DataFrame({'rel_size_z':[-1], 'loc_adv_z':[0]})).iloc[0]
pred_loc_neg = model.predict(pd.DataFrame({'rel_size_z':[0], 'loc_adv_z':[-1]})).iloc[0]

# Save outputs for downstream
print('n_rows', df.shape[0])
print('rel_size_mean', df['rel_size'].mean(), 'rel_size_std', df['rel_size'].std(ddof=0))
print('loc_adv_mean', df['loc_adv'].mean(), 'loc_adv_std', df['loc_adv'].std(ddof=0))
print('\nMain model coefficients')
print(summary)
print('\nInteraction model coefficients')
print(summary_int)
print('\nPredicted probabilities (main model)')
print({
    'baseline': baseline,
    '+1sd_rel_size': pred_rel,
    '-1sd_rel_size': pred_rel_neg,
    '+1sd_loc_adv': pred_loc,
    '-1sd_loc_adv': pred_loc_neg
})

# Compute pseudo R^2 (McFadden)
llf = model.llf
llnull = model.null
pseudo_r2 = 1 - llf/llnull
print('\nMcFadden R2', pseudo_r2)

