import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
path = 'crofoot.csv'

df = pd.read_csv(path)

# Define variables
# win: 1 if focal won
win = df['feature4'].astype(int)
# relative group size (focal - other)
rel_group_size = df['feature7'] - df['feature8']
# contest location advantage: other distance - focal distance (positive => closer to focal)
location_adv = df['feature6'] - df['feature5']

# Build dataframe for modeling
model_df = pd.DataFrame({
    'win': win,
    'rel_group_size': rel_group_size,
    'location_adv': location_adv
})

# Add constant
X = sm.add_constant(model_df[['rel_group_size', 'location_adv']])

# Fit logistic regression
logit_model = sm.Logit(model_df['win'], X)
result = logit_model.fit(disp=False)

# Compute odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf.columns = ['2.5%', '97.5%']
odds_ratios = np.exp(params)
conf_odds = np.exp(conf)

summary_df = pd.DataFrame({
    'coef': params,
    'odds_ratio': odds_ratios,
    'p_value': result.pvalues,
    'ci2.5': conf_odds['2.5%'],
    'ci97.5': conf_odds['97.5%']
})

# Also fit single-predictor models for robustness
X_size = sm.add_constant(model_df[['rel_group_size']])
X_loc = sm.add_constant(model_df[['location_adv']])

res_size = sm.Logit(model_df['win'], X_size).fit(disp=False)
res_loc = sm.Logit(model_df['win'], X_loc).fit(disp=False)

# Print key results
print('N:', len(model_df))
print('\nLogistic regression: win ~ rel_group_size + location_adv')
print(summary_df)
print('\nPseudo R2 (McFadden):', result.prsquared)

print('\nSingle predictor: rel_group_size')
print(res_size.summary2().tables[1][['Coef.', 'Std.Err.', 'P>|z|']])

print('\nSingle predictor: location_adv')
print(res_loc.summary2().tables[1][['Coef.', 'Std.Err.', 'P>|z|']])

# Compute simple descriptive stats
print('\nDescriptive stats:')
print(model_df.describe())

# Compute correlation for context
print('\nCorrelation matrix:')
print(model_df.corr())
