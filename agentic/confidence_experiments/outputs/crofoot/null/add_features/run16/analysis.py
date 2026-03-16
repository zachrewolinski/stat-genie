import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Keep relevant columns and drop rows with missing
cols = ['win','n_focal','n_other','dist_focal','dist_other']
df = _df[cols].dropna().copy()

# Derived variables
# Relative group size: focal minus other
# Relative location: other distance minus focal distance (positive => contest closer to focal home range center)
df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_loc'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for comparability
for c in ['rel_size','rel_loc']:
    df[c + '_z'] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

# Logistic regression
model = smf.glm('win ~ rel_size_z + rel_loc_z', data=df, family=sm.families.Binomial())
res = model.fit()

# Also check separate models for each predictor
model_size = smf.glm('win ~ rel_size_z', data=df, family=sm.families.Binomial())
res_size = model_size.fit()
model_loc = smf.glm('win ~ rel_loc_z', data=df, family=sm.families.Binomial())
res_loc = model_loc.fit()

# Basic summaries
summary = res.summary2().tables[1]
summary_size = res_size.summary2().tables[1]
summary_loc = res_loc.summary2().tables[1]

# Odds ratios for standardized predictors
or_full = np.exp(summary[['Coef.']])

# Predicted probabilities at +/-1 SD for each predictor holding other at mean
mean_row = df[['rel_size_z','rel_loc_z']].mean()

# Create scenarios
scenarios = pd.DataFrame([
    {'rel_size_z': -1, 'rel_loc_z': mean_row['rel_loc_z']},
    {'rel_size_z': 1, 'rel_loc_z': mean_row['rel_loc_z']},
    {'rel_size_z': mean_row['rel_size_z'], 'rel_loc_z': -1},
    {'rel_size_z': mean_row['rel_size_z'], 'rel_loc_z': 1},
])
scenarios = sm.add_constant(scenarios, has_constant='add')

preds = res.predict(scenarios)

# Save key results
print('N rows:', len(df))
print('\nFull model coefficients:')
print(summary)
print('\nOdds ratios (exp coef):')
print(or_full)

print('\nSize-only model:')
print(summary_size)
print('\nLocation-only model:')
print(summary_loc)

print('\nPredicted win prob at +/-1 SD:')
print('rel_size -1 SD:', preds.iloc[0])
print('rel_size +1 SD:', preds.iloc[1])
print('rel_loc -1 SD:', preds.iloc[2])
print('rel_loc +1 SD:', preds.iloc[3])

# Quick descriptive: mean win by sign of rel_size and rel_loc
print('\nMean win by rel_size sign:')
print(df.groupby(df['rel_size']>0)['win'].mean())
print('\nMean win by rel_loc sign:')
print(df.groupby(df['rel_loc']>0)['win'].mean())
